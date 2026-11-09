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

CPU tests of the in-kernel Q / K L2 norm of the non-deterministic KCP GDN package
(``flashinfer.experimental.cake.cake_gdn_nondet_train_kcp``; ``use_qk_l2norm_in_kernel``), mirrored from the deterministic package's tests (FlashInfer #7817 / #8063):

* the pure-torch reference ``cake_l2n`` against an exact rational oracle (every FP32 step rounded from the exact
  value; subnormals, zero rows, overflow of the sum of squares, wide exponents), in the documented orders;
* the fake backend's three programs (``fwd_prep_l2n`` / ``bwd_dqkg_l2n`` / ``qk_apply``) against the reference and
  against the norm-off programs on the normalized rows;
* the host plumbing on the fake backend: the composed chain (helper -> norm-off route on ``(q_hat, k_hat)`` ->
  helper pullback) bitwise, every saved / replay / checkpoint / handle form bitwise the direct one, ``qn`` / ``kn``
  never retained, the saved set v2, the option keyword, the byte queries; and the deterministic package with the
  norm on the same inputs within the deterministic accuracy constants;
* issue #8367 (norm off and on): plain ``split`` / ``view`` head views of a packed QKV row keep torch's autograd
  path -- their hooks and ``retain_grad`` fire, the gradients are bitwise (fake backend) those of contiguous
  operands and of ``cake_gdn_nondet_train_kcp_packed_qkv_views`` views, and only the packed views hand the packed dQKV row on without a
  concatenation.

The fake backend is not bitwise the device programs in #1 / #8's bodies (different accumulation orders): these tests
prove the plumbing and the oracle, not GPU numerics (``test_cake_gdn_nondet_train_kcp_dist.py`` holds the GPU rows).
The suite imports the package and the deterministic package through ``tests/cake/test_helpers/cake_package.py`` and nothing
else from flashinfer; the facade's hand-off of the option is tested in ``test_cake_gdn_nondet_train_kcp_api.py``.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import struct
import sys
from fractions import Fraction

import pytest
import torch


def _mod(sub: str | None = None):
    """The package (or one of its modules) through the test-helper registry: a stale entry fails here, never skips."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_nondet_train_kcp", sub)


def _det_mod(sub: str | None = None):
    """A module of the deterministic package through the test-helper registry (a stale entry fails, never skips)."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_det_train_kcp", sub)


kcp = _mod("cake_kcp")
plan = _mod("cake_kcp_plan")
l2n = _mod("cake_l2n")
fake = _mod("cake_fake_backend")
gbytes = _mod("cake_bytes")
jit = _mod("cake_jit")
CPU = torch.device("cpu")
GRADS = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd order of the leaves (q, k, v, g, beta)
TENSORS = ("out",) + GRADS
F32, BF16 = torch.float32, torch.bfloat16


@pytest.fixture(autouse=True)
def _no_flush_denormal():
    """The reference's policy is FTZ off: the tests run with torch's flush-denormal mode off (the default)."""
    torch.set_flush_denormal(False)
    assert not l2n.flush_denormal_active()
    yield


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (see ``test_cake_gdn_nondet_train_kcp.py``), so this module runs with at most 8
    threads and oneDNN off, as that suite does."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


# ----------------------------------------------------------------------------------------------------------------
# exact rational oracle of FP32 / BF16 round-to-nearest-even (subnormals, overflow, signed zeros)


def _pow2(e: int) -> Fraction:
    return Fraction(2) ** e


def _round(v: Fraction, mant: int) -> Fraction | None:
    """RN-even of a nonzero rational to a binary format with ``mant`` fraction bits and FP32's exponent range;
    ``None`` on overflow."""
    a = abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if _pow2(e) > a:
        e -= 1
    e = max(e, -126)
    q = _pow2(e - mant)
    n = a / q
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    res = fl * q
    if res >= _pow2(128):
        return None
    return res if v > 0 else -res


def _rn(v: Fraction, zero_sign: float = 1.0, mant: int = 23) -> float:
    """``v`` rounded to FP32 (``mant=23``) or BF16 (``mant=7``) as a Python float; ``zero_sign``: the sign of an exact
    zero result (IEEE)."""
    if v == 0:
        return math.copysign(0.0, zero_sign)
    r = _round(v, mant)
    if r is None:
        return math.copysign(math.inf, float(v))
    if r == 0:  # underflow to zero keeps the sign of the exact value
        return math.copysign(0.0, float(v))
    return float(r)


def _fr(x: float) -> Fraction:
    return Fraction(x)


def f_mul(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return (
            a * b
        )  # an infinite operand: the double product is the FP32 one (inf, nan or a signed zero never)
    return _rn(_fr(a) * _fr(b), zero_sign=math.copysign(1.0, a) * math.copysign(1.0, b))


def f_add(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return a + b
    s = _fr(a) + _fr(b)
    if s == 0:
        both_neg = math.copysign(1.0, a) < 0 and math.copysign(1.0, b) < 0
        return -0.0 if both_neg else 0.0
    return _rn(s)


def f_sub(a: float, b: float) -> float:
    return f_add(a, -b)


def f_fma(a: float, b: float, c: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return a * b + c
    s = _fr(a) * _fr(b) + _fr(c)
    if s == 0:
        p_neg = (math.copysign(1.0, a) * math.copysign(1.0, b)) < 0
        both_neg = p_neg and math.copysign(1.0, c) < 0
        return -0.0 if both_neg else 0.0
    return _rn(s)


def f_div(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return a / b if b != 0 else math.copysign(math.inf, a) * math.copysign(1.0, b)
    return _rn(_fr(a) / _fr(b), zero_sign=math.copysign(1.0, a) * math.copysign(1.0, b))


def _next_f32(x: float, up: bool) -> float:
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = bits + 1 if (up == (x >= 0)) else bits - 1
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f_sqrt(a: float) -> float:
    """Correctly rounded FP32 square root of a non-negative FP32 value, decided by exact midpoints."""
    if a == 0 or not math.isfinite(a):
        return math.sqrt(a)
    s = _rn(_fr(math.sqrt(a)))
    fa = _fr(a)
    for _ in range(4):
        lo = (_fr(_next_f32(s, False)) + _fr(s)) / 2 if s > 0 else Fraction(0)
        hi = (_fr(s) + _fr(_next_f32(s, True))) / 2
        if lo * lo < fa < hi * hi:
            return s
        s = _next_f32(s, fa > hi * hi)
    raise AssertionError("sqrt oracle did not converge")


def _bits32(t: torch.Tensor) -> list[int]:
    return t.contiguous().view(torch.int32).flatten().tolist()


def _bits16(t: torch.Tensor) -> list[int]:
    return t.contiguous().view(torch.int16).flatten().tolist()


def _f32_list(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.to(F32).flatten().tolist()]


def oracle_forward(row: list[float]) -> tuple[float, float, list[float]]:
    """``(ss, r, q_hat)`` of one 128-channel row (the device order of ``cake_l2n``)."""
    lanes = []
    for c in range(16):
        x = row[8 * c : 8 * c + 8]
        s = f_fma(x[0], x[0], 0.0)
        for k in range(1, 8):
            s = f_fma(x[k], x[k], s)
        lanes.append(s)
    h = 8
    while h >= 1:
        lanes = [f_add(lanes[i], lanes[i + h]) for i in range(h)]
        h //= 2
    ss = lanes[0]
    eps = struct.unpack("<f", struct.pack("<f", 1e-6))[0]
    r = f_div(1.0, f_sqrt(f_add(ss, eps)))
    q_hat = []
    for x in row:
        p = f_mul(x, r)
        q_hat.append(
            p
            if not math.isfinite(p)
            else _rn(_fr(p), zero_sign=math.copysign(1.0, p), mant=7)
        )
    return ss, r, q_hat


def oracle_pullback(row: list[float], r: float, D: list[float]) -> list[float]:
    y = [f_mul(x, r) for x in row]
    p = [f_mul(d, yy) for d, yy in zip(D, y, strict=True)]
    S = []
    for qd in range(4):
        a = p[32 * qd : 32 * qd + 32]
        h = 16
        while h >= 1:
            a = [f_add(a[i], a[i + h]) for i in range(h)]
            h //= 2
        S.append(a[0])
    dot = f_add(f_add(f_add(S[0], S[1]), S[2]), S[3])
    return [f_mul(r, f_sub(d, f_mul(yy, dot))) for d, yy in zip(D, y, strict=True)]


def _edge_rows() -> torch.Tensor:
    """BF16 rows ``[n, 128]``: random scales, a zero row, an overflowing sum of squares, subnormal rows, wide
    exponents in one row, a sum near the FP32 normal bound, signed zeros."""
    gen = torch.Generator().manual_seed(78170)
    rows = [
        torch.randn(128, generator=gen) * s
        for s in (1.0, 1e-3, 37.0, 2.0**30, 2.0**-30)
    ]
    rows.append(torch.zeros(128))
    rows.append(
        torch.full((128,), 2.0**70)
    )  # x^2 = 2^140 overflows FP32: ss = inf, r = 0
    big = torch.randn(128, generator=gen)
    big[5] = 2.0**64  # x^2 = 2^128: the first overflowing square
    rows.append(big)
    rows.append(torch.randn(128, generator=gen) * 2.0**-130)  # subnormal BF16
    rows.append(
        torch.randn(128, generator=gen) * 2.0**-70
    )  # squares in FP32's subnormal range
    rows.append(torch.randn(128, generator=gen) * 2.0**-63)  # ss near 2^-126
    wide = torch.randn(128, generator=gen)
    wide[::3] *= 2.0**40
    wide[1::3] *= 2.0**-40
    wide[2::7] *= 2.0**-128
    rows.append(wide)
    signed = torch.zeros(128)
    signed[::2] = -0.0
    signed[7] = -3.0
    rows.append(signed)
    return torch.stack(rows).to(BF16)


# ----------------------------------------------------------------------------------------------------------------
# the reference against the oracle


def test_ss_fma_emulation_is_the_exact_rounding() -> None:
    """The FP64 emulation of ``fma.rn(x, x, s)`` equals the FP32 rounding of the exact ``x^2 + s`` (and the whole
    forward the oracle's) on every edge row, bit for bit."""
    rows = _edge_rows()
    ss = l2n.sum_of_squares(rows.to(F32))
    q_hat, r = l2n.qk_l2norm_ref(rows)
    for i in range(rows.shape[0]):
        row = _f32_list(rows[i])
        o_ss, o_r, o_q = oracle_forward(row)
        assert _bits32(ss[i : i + 1]) == _bits32(torch.tensor([o_ss], dtype=F32)), (
            i,
            float(ss[i]),
            o_ss,
        )
        assert _bits32(r[i : i + 1]) == _bits32(torch.tensor([o_r], dtype=F32)), (
            i,
            float(r[i]),
            o_r,
        )
        assert _bits16(q_hat[i]) == _bits16(torch.tensor(o_q, dtype=F32).to(BF16)), i
    # the overflow row: r = 0 exactly; the zero row: r = 1 / sqrt(fp32(1e-6))
    assert float(r[6]) == 0.0 and math.isinf(float(ss[6]))
    assert float(r[5]) == f_div(
        1.0, f_sqrt(struct.unpack("<f", struct.pack("<f", 1e-6))[0])
    )


def test_ss_fma_emulation_random_exponent_sweep() -> None:
    """Each fma step of the emulation against the exact rounding, over random BF16 pairs with exponents across the
    whole range (subnormal squares and overflowing ones included)."""
    gen = torch.Generator().manual_seed(7817)
    x = (
        torch.randn(4096, generator=gen)
        * torch.exp2(torch.randint(-140, 70, (4096,), generator=gen).float())
    ).to(BF16)
    s = (
        torch.rand(4096, generator=gen)
        * torch.exp2(torch.randint(-149, 128, (4096,), generator=gen).float())
    ).to(F32)
    xd = x.to(F32).to(torch.float64)
    got = (xd * xd + s.to(torch.float64)).to(F32)
    for i in range(0, 4096, 7):
        a, b = float(x[i].float()), float(s[i])
        want = f_fma(a, a, b)
        assert _bits32(got[i : i + 1]) == _bits32(torch.tensor([want], dtype=F32)), (
            a,
            b,
        )


def test_pullback_reference_is_the_oracle() -> None:
    """``l2n_pullback`` equals the oracle's ``r (D - y dot)`` in the documented order on the edge rows, with random
    and edge gradients (zeros, wide magnitudes)."""
    rows = _edge_rows()
    _q, r = l2n.qk_l2norm_ref(rows)
    gen = torch.Generator().manual_seed(78171)
    D = torch.randn(rows.shape, generator=gen) * torch.exp2(
        torch.randint(-60, 60, rows.shape, generator=gen).float()
    )
    D[3] = 0.0
    D[4, ::5] = -0.0
    got = l2n.l2n_pullback(rows, r, D)
    for i in range(rows.shape[0]):
        want = oracle_pullback(_f32_list(rows[i]), float(r[i]), _f32_list(D[i]))
        w = torch.tensor(want, dtype=F32)
        assert _bits32(got[i]) == _bits32(w), i


def test_reference_takes_strided_rows_and_the_apply_form() -> None:
    """A packed-QKV view gives the contiguous form's bits; ``qk_apply_ref`` on the forward's ``r`` is its
    ``q_hat``."""
    gen = torch.Generator().manual_seed(5)
    T, H = 37, 3
    buf = (torch.randn(T, (2 * H + 6) * 128, generator=gen) * 4).to(BF16)
    q = buf[:, : H * 128].view(T, H, 128)
    hat, r = l2n.qk_l2norm_ref(q)
    hat_c, r_c = l2n.qk_l2norm_ref(q.contiguous())
    assert torch.equal(hat, hat_c) and torch.equal(r, r_c)
    assert hat.shape == (T, H, 128) and r.shape == (T, H) and hat.is_contiguous()
    assert torch.equal(l2n.qk_apply_ref(q, r), hat)


# ----------------------------------------------------------------------------------------------------------------
# helpers of the host tests


def _raw_inputs(
    cu: list[int], H_qk: int, HV: int, seed: int = 20261029
) -> dict[str, torch.Tensor]:
    """Seeded CPU operands with RAW q / k (per-row magnitudes from 2^-8 to 2^8): the in-kernel norm's input."""
    gen = torch.Generator().manual_seed(seed)
    T = cu[-1]
    a_log = torch.rand(HV, generator=gen) * math.log(16.0)

    def raw(H):
        scale = torch.exp2(torch.randint(-8, 9, (T, H, 1), generator=gen).float())
        return (torch.randn(T, H, 128, generator=gen) * scale).to(BF16)

    return {
        "q": raw(H_qk),
        "k": raw(H_qk),
        "v": torch.randn(T, HV, 128, generator=gen).to(BF16),
        "g": (
            -torch.exp(a_log)[None, :]
            * torch.nn.functional.softplus(torch.randn(T, HV, generator=gen))
        )
        .float()
        .contiguous(),
        "beta": torch.sigmoid(torch.randn(T, HV, generator=gen)).float().contiguous(),
        "dout": torch.randn(T, HV, 128, generator=gen).to(BF16),
    }


CU = [0, 100, 740, 817, 1000]


def _setup(H_qk: int = 2, HV: int = 6, cu=CU):
    part = plan.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    return ctx, _raw_inputs(cu, H_qk, HV)


def _ops(ops):
    return tuple(ops[n] for n in ("q", "k", "v", "g", "beta"))


def _saved_pass(ctx, ops, *, q=None, k=None, ws=None, **o) -> dict[str, torch.Tensor]:
    qq = ops["q"] if q is None else q
    kk = ops["k"] if k is None else k
    o = {"backend": "fake", **o}
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
        qq,
        kk,
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        workspace=ws,
        **o,
    )
    dq, dk, dv, dbeta, dg = kcp.cake_gdn_nondet_train_kcp_backward(
        qq,
        kk,
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        ops["dout"],
        saved=saved,
        workspace=ws,
        **o,
    )
    return {
        "out": out,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dbeta": dbeta,
        "dg": dg,
        "saved": saved,
    }


def _autograd(
    ctx, ops, *, ckpt: bool, handle: bool = False, twice: bool = False, **o
) -> dict[str, torch.Tensor]:
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    o = {"backend": "fake", **o}
    hd = kcp.CakeGDNNondetKcpRecomputeHandle() if handle else None

    def fn(q, k, v, g, beta, h):
        return kcp.cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, recompute_handle=h, **o)

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, *leaves, hd, use_reentrant=False)
    else:
        out = fn(*leaves, hd)
    out.backward(ops["dout"], retain_graph=twice)
    if twice:
        out.backward(ops["dout"])
    return {
        "out": out.detach(),
        **{n: lf.grad for n, lf in zip(GRADS, leaves, strict=True)},
    }


def _same(a: dict, b: dict, what: str, names=TENSORS) -> None:
    for n in names:
        assert torch.equal(a[n], b[n]), f"{what}: {n} differs"


def _norm_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return l2n.qk_l2norm_ref(x)


# ----------------------------------------------------------------------------------------------------------------
# the fake programs


def _local_call(
    ctx, ops, *, l2n_on: bool, q=None, k=None, backward=False, group_reduce="fused"
):
    opts = kcp.make_options(
        backend="fake", use_qk_l2norm_in_kernel=l2n_on, group_reduce=group_reduce
    )
    call, _ = kcp._prepare(
        ops["q"] if q is None else q,
        ops["k"] if k is None else k,
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        scale=None,
        opts=opts,
        workspace=kcp.CakeGDNNondetKcpWorkspace(),
        backward=backward,
        dout=ops["dout"] if backward else None,
    )
    # zero-initialised tapes: positions no program writes compare equal across the two runs
    for name in kcp.FWD_TAPES:
        call.tapes[name] = call.tape(name).zero_()
    if l2n_on:
        for name in (*kcp.QK_TAPES, kcp.QK_RSTD):
            call.tapes[name] = call.tape(name).zero_()
    return call


PREP_TAPES = ("w", "u", "aqk", "akk", "g_cum", "rowscale", "qscale", "gamma")


def test_fake_fwd_prep_l2n_is_the_reference_then_v1() -> None:
    """``fwd_prep_l2n``'s ``qn`` / ``kn`` / ``r`` are the reference's, its eight tapes those of ``fwd_prep`` run on
    the materialized ``(qn, kn)``, and ``qk_apply`` on its ``r`` rebuilds its ``qn`` / ``kn`` bitwise."""
    ctx, ops = _setup()
    c_on = _local_call(ctx, ops, l2n_on=True)
    c_on.local("fwd_prep")
    assert c_on.sequence == ["fwd_prep_l2n"]
    qn, kn, rstd = (c_on.tape(n) for n in ("qn", "kn", kcp.QK_RSTD))
    for x, xn, plane in ((ops["q"], qn, 0), (ops["k"], kn, 1)):
        hat, r = l2n.qk_l2norm_ref(x)
        assert torch.equal(xn, hat) and torch.equal(rstd[plane], r)
    c_off = _local_call(ctx, ops, l2n_on=False, q=qn.clone(), k=kn.clone())
    c_off.local("fwd_prep")
    for n in PREP_TAPES:
        assert torch.equal(c_on.tape(n), c_off.tape(n)), n
    c_apply = _local_call(ctx, ops, l2n_on=True, backward=True)
    c_apply.bufs[kcp.QK_RSTD] = rstd.clone()
    c_apply.local("qk_apply")
    assert c_apply.sequence == ["qk_apply"]
    assert torch.equal(c_apply.tape("qn"), qn) and torch.equal(c_apply.tape("kn"), kn)


def test_fake_qk_apply_takes_strided_raw_rows() -> None:
    """Packed-QKV raw views (token stride ``(2 H_qk + HV) * 128``) reach ``qk_apply`` / ``fwd_prep_l2n`` as such and
    give the contiguous operands' bits."""
    ctx, ops = _setup()
    T, H = ops["q"].shape[0], ops["q"].shape[1]
    HV = ops["v"].shape[1]
    buf = torch.cat(
        [ops["q"].reshape(T, -1), ops["k"].reshape(T, -1), ops["v"].reshape(T, -1)], -1
    ).contiguous()
    qs, ks = (
        buf[:, : H * 128].view(T, H, 128),
        buf[:, H * 128 : 2 * H * 128].view(T, H, 128),
    )
    assert qs.stride(0) == (2 * H + HV) * 128
    a = _local_call(ctx, ops, l2n_on=True)
    a.local("fwd_prep")
    b = _local_call(ctx, ops, l2n_on=True, q=qs, k=ks)
    b.local("fwd_prep")
    for n in (*PREP_TAPES, "qn", "kn", kcp.QK_RSTD):
        assert torch.equal(a.tape(n), b.tape(n)), n


@pytest.mark.parametrize("heads", [(2, 2), (2, 6)])
@pytest.mark.parametrize("group_reduce", ["fused", "host"])
def test_end_to_end_is_the_composed_chain(heads, group_reduce) -> None:
    """Flag on == the composed chain, bitwise: ``out`` / ``dv`` / ``dbeta`` / ``dg`` are the norm-off route's on the
    materialized ``(q_hat, k_hat)``; ``dq`` / ``dk`` are ``bf16(l2n_pullback(x, r, D))`` with ``D`` the norm-off
    route's FP32 group sum of the per-value-head partials (``group_reduce="host"``: the head-order sum; ``"fused"`` at
    one value head per Q/K head, where the fused sum is that partial)."""
    H_qk, HV = heads
    if group_reduce == "fused" and H_qk != HV:
        pytest.skip(
            "the fused FP32 group sum of the emulation is not the head-order sum of its partials"
        )
    ctx, ops = _setup(H_qk, HV)
    on = _saved_pass(ctx, ops, use_qk_l2norm_in_kernel=True, group_reduce=group_reduce)
    qh, rq = l2n.qk_l2norm_ref(ops["q"])
    kh, rk = l2n.qk_l2norm_ref(ops["k"])
    ws = kcp.CakeGDNNondetKcpWorkspace()
    off = _saved_pass(ctx, ops, q=qh, k=kh, ws=ws, group_reduce="host")
    _same(on, off, "on vs off on (q_hat, k_hat)", names=("out", "dv", "dbeta", "dg"))
    T, r = ops["q"].shape[0], HV // H_qk
    for name, part, x, rr in (
        ("dq", "dq_part", ops["q"], rq),
        ("dk", "dk_part", ops["k"], rk),
    ):
        p = ws.tape(part).view(T, H_qk, r, 128)
        acc = p[:, :, 0]
        for j in range(1, r):
            acc = acc + p[:, :, j]
        want = l2n.l2n_pullback(x, rr, acc).to(BF16)
        assert torch.equal(on[name], want), name
    assert torch.equal(on["saved"].qk_rstd, torch.stack([rq, rk]))


def test_host_pullback_block_split_is_bitwise() -> None:
    """The host reduction's token blocks change nothing: one block == many blocks."""
    ctx, ops = _setup(2, 6)
    base = _saved_pass(ctx, ops, use_qk_l2norm_in_kernel=True, group_reduce="host")
    old = kcp.HOST_L2N_BLOCK_TOKENS
    try:
        kcp.HOST_L2N_BLOCK_TOKENS = 64
        small = _saved_pass(ctx, ops, use_qk_l2norm_in_kernel=True, group_reduce="host")
    finally:
        kcp.HOST_L2N_BLOCK_TOKENS = old
    _same(base, small, "host blocks")


# ----------------------------------------------------------------------------------------------------------------
# mode equalities (plumbing: the fake is deterministic, so every form must reproduce the direct bits)


@pytest.mark.parametrize("segments", [1, 2])
def test_modes_are_bitwise_with_the_norm(segments) -> None:
    """light == full (``qk_apply``) == replay mode == ``save_transfer`` False == autograd == checkpoint (replay form,
    kept tapes, ``"auto"``) == recompute handle == the copy layout == packed views, with the norm on; and the
    norm changes the result."""
    ctx, ops = _setup()
    o = dict(use_qk_l2norm_in_kernel=True, segments=segments)
    ref = _saved_pass(ctx, ops, **o)
    assert (
        kcp.LAST_CALL["bwd"]["qk_source"] == "replay"
        and kcp.LAST_CALL["bwd"]["rstd_source"] == "saved"
    )
    _same(ref, _saved_pass(ctx, ops, **o), "repeat")
    _same(ref, _saved_pass(ctx, ops, save_mode="full", **o), "full")
    assert kcp.LAST_CALL["bwd"]["qk_source"] == "qk_apply"
    assert "qk_apply" in kcp.LAST_CALL["bwd"]["sequence"]
    _same(ref, _saved_pass(ctx, ops, boundary_mode="replay", **o), "replay mode")
    assert (
        kcp.LAST_CALL["bwd"]["qk_source"] == "recompute"
        and kcp.LAST_CALL["bwd"]["rstd_source"] == "arena"
    )
    _same(ref, _saved_pass(ctx, ops, save_transfer=False, **o), "save_transfer=False")
    _same(ref, _saved_pass(ctx, ops, layout="copy", **o), "layout=copy")
    direct = _autograd(ctx, ops, ckpt=False, **o)
    _same(ref, direct, "autograd")
    for reuse in (False, True, "auto"):
        stats = dict(kcp.KEPT_STATS)
        got = _autograd(ctx, ops, ckpt=True, reuse_recompute_tapes=reuse, **o)
        _same(direct, got, f"checkpoint reuse={reuse}")
        if reuse is True:
            assert kcp.KEPT_STATS["take"] - stats["take"] == 1
            assert kcp.LAST_CALL["bwd"]["tapes_source"] == "kept"
        if kcp.LAST_CALL["bwd"]["tapes_source"] == "kept":
            # the recompute forward's qn / kn came with the tapes: no program re-reads the raw rows to normalize
            assert kcp.LAST_CALL["bwd"]["qk_source"] == "kept"
            assert not {"qk_apply", "fwd_prep_l2n"} & set(
                kcp.LAST_CALL["bwd"]["sequence"]
            )
    _same(
        direct, _autograd(ctx, ops, ckpt=True, save_mode="full", **o), "checkpoint full"
    )
    _same(
        direct,
        _autograd(ctx, ops, ckpt=True, boundary_mode="replay", **o),
        "checkpoint replay",
    )
    _same(
        direct,
        _autograd(ctx, ops, ckpt=True, handle=True, save_transfer=False, **o),
        "handle",
    )
    twice = _autograd(
        ctx, ops, ckpt=True, handle=True, twice=True, save_transfer=False, **o
    )
    for n in GRADS:
        assert torch.equal(twice[n], direct[n] + direct[n]), n
    # packed QKV views (the convolution's output): bitwise, and the split still aliases the packed gradient row
    T, H, HV = ops["q"].shape[0], ops["q"].shape[1], ops["v"].shape[1]
    qkv = (
        torch.cat([ops[n].reshape(T, -1) for n in ("q", "k", "v")], -1)
        .contiguous()
        .requires_grad_(True)
    )
    g, beta = (ops[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    before = dict(kcp.PACKED_SPLIT_STATS)
    pq, pk, pv = kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(qkv, H_qk=H, HV=HV)
    out = kcp.cake_gdn_nondet_kcp(pq, pk, pv, g, beta, ctx, backend="fake", **o)
    out.backward(ops["dout"])
    assert kcp.PACKED_SPLIT_STATS["aliased"] - before["aliased"] == 1
    assert kcp.LAST_CALL["bwd"]["grad_layout"] == "packed"
    grad = qkv.grad
    packed = {
        "out": out.detach(),
        "dq": grad[:, : H * 128].reshape(T, H, 128),
        "dk": grad[:, H * 128 : 2 * H * 128].reshape(T, H, 128),
        "dv": grad[:, 2 * H * 128 :].reshape(T, HV, 128),
        "dg": g.grad,
        "dbeta": beta.grad,
    }
    _same(direct, packed, "packed views")
    off = _saved_pass(ctx, ops, segments=segments)
    assert not torch.equal(off["out"], ref["out"]) and not torch.equal(
        off["dq"], ref["dq"]
    )


def _split_view_pass(ctx, ops, form: str, *, ckpt: bool, **o) -> dict:
    """One autograd pass on q / k / v as head views of a packed ``[T, (2 H_qk + HV) * 128]`` leaf.  ``form``:
    ``"plain"`` = ``split`` + ``view`` (torch's own split backward), ``"packed"`` = :func:`cake_gdn_nondet_train_kcp_packed_qkv_views`,
    ``"replace"`` = packed views with a hook on ``q`` that returns a new gradient.  A tensor hook (recording the
    gradient) and ``retain_grad`` are set on every view (and on the split outputs) of the original forward."""
    T, H, HV = ops["q"].shape[0], ops["q"].shape[1], ops["v"].shape[1]
    row = torch.cat([ops[n].reshape(T, -1) for n in ("q", "k", "v")], -1)
    row = row.contiguous().requires_grad_(True)
    g, beta = (ops[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    fired: dict[str, list[torch.Tensor]] = {}
    views: dict[str, torch.Tensor] = {}

    def fn(row, g, beta):
        if form == "plain":
            parts = row.split((H * 128, H * 128, HV * 128), dim=-1)
            q, k, v = (
                p.view(T, n, 128) for p, n in zip(parts, (H, H, HV), strict=True)
            )
        else:
            parts = ()
            q, k, v = kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(row, H_qk=H, HV=HV)
        if not views:  # the original forward (a checkpoint recompute builds a graph no backward runs)
            views.update(q=q, k=k, v=v, **{f"part{i}": p for i, p in enumerate(parts)})
            for name, t in views.items():
                t.register_hook(
                    lambda gr, name=name: fired.setdefault(name, []).append(gr.clone())
                )
                t.retain_grad()
            if form == "replace":
                q.register_hook(lambda gr: gr.clone())
        return kcp.cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, backend="fake", **o)

    before = dict(kcp.PACKED_SPLIT_STATS)
    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, row, g, beta, use_reentrant=False)
    else:
        out = fn(row, g, beta)
    out.backward(ops["dout"])
    return {
        "out": out.detach(),
        "row": row.grad,
        "dg": g.grad,
        "dbeta": beta.grad,
        "fired": fired,
        "views": views,
        "grad_layout": kcp.LAST_CALL["bwd"]["grad_layout"],
        "split": {n: kcp.PACKED_SPLIT_STATS[n] - before[n] for n in before},
    }


@pytest.mark.parametrize("ckpt", [False, True], ids=["direct", "checkpoint"])
@pytest.mark.parametrize("norm", [False, True], ids=["l2n-off", "l2n-on"])
def test_plain_split_views_keep_hooks_and_bitwise_gradients(norm, ckpt) -> None:
    """Issue #8367 (fake backend): plain ``split`` / ``view`` head views of one packed QKV row take torch's own
    autograd path (the GDN backward writes three contiguous ``dq`` / ``dk`` / ``dv``, the split backward concatenates
    them): tensor hooks and ``retain_grad`` on ``q`` / ``k`` / ``v`` and on the split outputs fire once with the
    contiguous call's gradients, and the row gradient, ``dg``, ``dbeta`` and ``out`` are bitwise those of contiguous
    operands and of :func:`cake_gdn_nondet_train_kcp_packed_qkv_views` views.  Only the packed views get one packed dQKV row, which their
    split hands on; hooks on the packed views fire too, and a hook that replaces a gradient makes that split
    concatenate (same values)."""
    ctx, ops = _setup()
    if not norm:
        # unit-norm Q / K rows: valid operands with the norm off (the raw rows are the norm-on operands)
        for n in ("q", "k"):
            x = ops[n].float()
            ops[n] = (x / x.norm(dim=-1, keepdim=True)).to(BF16)
    o = {"use_qk_l2norm_in_kernel": norm}
    ref = _autograd(ctx, ops, ckpt=ckpt, **o)
    T = ops["q"].shape[0]
    want = {n: ref["d" + n] for n in ("q", "k", "v")}
    row = torch.cat([want[n].reshape(T, -1) for n in ("q", "k", "v")], -1)
    expect_split = {
        "plain": {"aliased": 0, "concatenated": 0},
        "packed": {"aliased": 1, "concatenated": 0},
        "replace": {"aliased": 0, "concatenated": 1},
    }
    for form, split in expect_split.items():
        got = _split_view_pass(ctx, ops, form, ckpt=ckpt, **o)
        assert torch.equal(got["out"], ref["out"]), form
        assert torch.equal(got["row"], row), form
        assert torch.equal(got["dg"], ref["dg"]) and torch.equal(
            got["dbeta"], ref["dbeta"]
        ), form
        # the GDN backward writes one packed dQKV row only for the packed views (whose split hands it on, or
        # concatenates it after a replacing hook); plain views get three contiguous gradients
        assert got["grad_layout"] == ("separate" if form == "plain" else "packed"), form
        assert got["split"] == split, form
        assert sorted(got["fired"]) == sorted(got["views"]), form
        for i, n in enumerate(("q", "k", "v")):
            assert len(got["fired"][n]) == 1, (form, n)
            assert torch.equal(got["fired"][n][0], want[n]), (form, n)
            assert torch.equal(got["views"][n].grad, want[n]), (form, n)
            if form == "plain":
                part = got["fired"][f"part{i}"]
                assert len(part) == 1 and torch.equal(part[0], want[n].reshape(T, -1))
                pg = got["views"][f"part{i}"].grad
                assert torch.equal(pg, want[n].reshape(T, -1)), (form, n)


def test_packed_views_refuse_in_place_changes() -> None:
    """The packed views are outputs of one custom autograd node that are views of ``qkv``: autograd refuses an
    in-place change to them while ``qkv`` requires grad (the README's packed Q/K/V note)."""
    H, HV, T = 2, 6, 5
    qkv = torch.zeros(T, (2 * H + HV) * 128, dtype=BF16, requires_grad=True)
    q, k, v = kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(qkv, H_qk=H, HV=HV)
    for t in (q, k, v):
        with pytest.raises(RuntimeError, match="inplace"):
            t.mul_(2)


def test_norm_off_is_the_v1_route() -> None:
    """With the flag off nothing of the norm runs or is bound: the v1 program sequence, no qn / kn / rstd tape,
    the v1 saved set (no ``qk_rstd``), and ``LAST_CALL`` says so."""
    ctx, ops = _setup()
    ws = kcp.CakeGDNNondetKcpWorkspace()
    res = _saved_pass(ctx, ops, ws=ws)
    for kind in ("fwd", "bwd"):
        rec = kcp.LAST_CALL[kind]
        assert (
            rec["qk_l2norm"] is False
            and rec["qk_source"] is None
            and rec["rstd_source"] is None
        )
        assert not set(rec["sequence"]) & set(kcp.L2N_PROGRAMS), rec["sequence"]
    assert not set(ws._tapes) & {*kcp.QK_TAPES, kcp.QK_RSTD}
    saved = res["saved"]
    assert (
        saved.qk_rstd is None and saved.qk_l2norm is False and len(saved.flags()) == 3
    )
    assert saved.meta()["qk_l2norm"] is False
    assert all(t.dim() != 3 or t.shape[0] != 2 for t in saved.tensors())


# ----------------------------------------------------------------------------------------------------------------
# record_qk: backward q_hat / k_hat / r == the forward's, on every source


@pytest.mark.parametrize(
    "opts,source",
    [
        ({}, "replay"),
        ({"save_mode": "full"}, "qk_apply"),
        ({"boundary_mode": "replay"}, "recompute"),
    ],
)
def test_backward_rebuilds_the_forward_rows(opts, source) -> None:
    ctx, ops = _setup()
    knobs = {"test_knobs": {"record_qk": True}}
    o = dict(use_qk_l2norm_in_kernel=True, **opts, **knobs)
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
        *_ops(ops), ctx, backend="fake", **o
    )
    fwd = kcp.LAST_CALL["fwd"]["qk"]
    kcp.cake_gdn_nondet_train_kcp_backward(
        *_ops(ops),
        ctx,
        ops["dout"],
        saved=saved,
        backend="fake",
        **o,
    )
    bwd = kcp.LAST_CALL["bwd"]["qk"]
    assert kcp.LAST_CALL["bwd"]["qk_source"] == source
    qh, rq = l2n.qk_l2norm_ref(ops["q"])
    kh, rk = l2n.qk_l2norm_ref(ops["k"])
    assert torch.equal(fwd["qn"], qh) and torch.equal(fwd["kn"], kh)
    assert torch.equal(fwd["rstd"], torch.stack([rq, rk]))
    for n in ("qn", "kn", "rstd"):
        assert torch.equal(bwd[n], fwd[n]), n
    if source == "replay":
        assert bwd["rstd_regen_equal"] is True and torch.equal(
            bwd["rstd_regen"], saved.qk_rstd
        )


# ----------------------------------------------------------------------------------------------------------------
# lifetimes: qn / kn / r never outlive a call; the saved set grows by r only


def test_norm_tapes_are_not_retained() -> None:
    """After an eager forward the caller's arena holds no ``qn`` / ``kn`` / ``qk_rstd`` (``LAST_CALL`` records the
    arena), the saved set holds no ``[T, H_qk, 128]`` tensor and grows by ``8 T_r H_qk`` bytes exactly (the byte
    query's term), and the backward leaves none of them either."""
    ctx, ops = _setup()
    H_qk, HV = ops["q"].shape[1], ops["v"].shape[1]
    T = ops["q"].shape[0]
    for opts in (
        # the default with the norm is "light_akk" (its Akk tape: test_..._light_akk.py)
        {"save_mode": "light"},
        {"save_mode": "full"},
        {"boundary_mode": "replay"},
        {"save_for_backward": False},
    ):
        ws = kcp.CakeGDNNondetKcpWorkspace()
        out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
            *_ops(ops),
            ctx,
            backend="fake",
            workspace=ws,
            use_qk_l2norm_in_kernel=True,
            **opts,
        )
        bad = {*kcp.QK_TAPES, kcp.QK_RSTD}
        assert not set(ws._tapes) & bad, (opts, sorted(ws._tapes))
        assert not set(kcp.LAST_CALL["fwd"]["arena_tapes"]) & bad
        if saved is None:
            continue
        assert all(tuple(t.shape) != (T, H_qk, 128) for t in saved.tensors())
        _o, saved_off = kcp.cake_gdn_nondet_train_kcp_forward(
            *_ops(ops), ctx, backend="fake", **opts
        )
        assert saved.bytes - saved_off.bytes == 8 * T * H_qk
        terms = gbytes.cake_gdn_nondet_train_kcp_saved_layout(
            ctx,
            H_qk=H_qk,
            HV=HV,
            use_qk_l2norm_in_kernel=True,
            **{k: v for k, v in opts.items()},
        )
        assert terms["qk_rstd"] == 8 * T * H_qk
        assert (
            gbytes.cake_gdn_nondet_train_kcp_saved_bytes(
                ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=True, **opts
            )
            - (
                gbytes.cake_gdn_nondet_train_kcp_saved_bytes(
                    ctx, H_qk=H_qk, HV=HV, **opts
                )
            )
            == 8 * T * H_qk
        )
        kcp.cake_gdn_nondet_train_kcp_backward(
            *_ops(ops),
            ctx,
            ops["dout"],
            saved=saved,
            backend="fake",
            workspace=ws,
            use_qk_l2norm_in_kernel=True,
            **{k: v for k, v in opts.items() if k != "save_for_backward"},
        )
        assert not set(ws._tapes) & bad, (opts, sorted(ws._tapes))
    assert "qk_rstd" not in gbytes.cake_gdn_nondet_train_kcp_saved_layout(
        ctx, H_qk=H_qk, HV=HV
    )  # norm-off layout unchanged


@pytest.mark.parametrize("fused_o", (0, 1))
def test_forward_drops_kn_before_the_output_and_qn_after_fwd_o(fused_o: int) -> None:
    """The arena at #5 holds ``qn`` but no ``kn`` (the tape walks were its last readers); ``qn`` goes after #5.  The
    fused forward (``fused_o`` = 1, the shipped form) reads both rows in its output walks and drops both after them;
    either form leaves neither row in the arena."""
    ctx, ops = _setup()
    seen = {}
    ws = kcp.CakeGDNNondetKcpWorkspace()

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if not call.backward:
                seen[name] = sorted(set(call.ws._tapes) & {"qn", "kn", kcp.QK_RSTD})
            super().run(name, args)

    previous = kcp.set_backend("fake", Recording())
    try:
        kcp.cake_gdn_nondet_train_kcp_forward(
            *_ops(ops),
            ctx,
            backend="fake",
            workspace=ws,
            use_qk_l2norm_in_kernel=True,
            test_knobs={"fused_o": fused_o},
        )
    finally:
        kcp.set_backend("fake", previous)
    assert seen["fwd_prep_l2n"] == [
        "kn",
        "qn",
    ]  # the saved r is a fresh tensor, never the arena's
    if fused_o and kcp.fused_o_shipped():
        assert seen["kcp_fwdh_out"] == ["kn", "qn"]
        assert "kcp_fwdh_tape" not in seen and "fwd_o" not in seen
    else:
        assert seen["kcp_fwdh_tape"] == ["kn", "qn"]
        assert seen["fwd_o"] == ["qn"]
        assert "kcp_fwdh_out" not in seen
    assert not {"qn", "kn"} & set(ws._tapes)


@pytest.mark.parametrize("fused_o_tape", (0, 1))
def test_full_forward_reads_both_rows_in_the_tape_and_output_walk(
    fused_o_tape: int,
) -> None:
    """``save_mode="full"``: the shipped forward (``fused_o_tape`` = 1) reads ``qn`` and ``kn`` in its tape-and-output
    walks and drops both after them; the split form (0) reads both in the tape walk and ``qn`` in #5.  Either form
    leaves neither row in the arena (the saved ``r`` is a fresh tensor)."""
    ctx, ops = _setup()
    seen = {}
    ws = kcp.CakeGDNNondetKcpWorkspace()

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if not call.backward:
                seen[name] = sorted(set(call.ws._tapes) & {"qn", "kn", kcp.QK_RSTD})
            super().run(name, args)

    previous = kcp.set_backend("fake", Recording())
    try:
        kcp.cake_gdn_nondet_train_kcp_forward(
            *_ops(ops),
            ctx,
            backend="fake",
            workspace=ws,
            save_mode="full",
            use_qk_l2norm_in_kernel=True,
            test_knobs={"fused_o_tape": fused_o_tape},
        )
    finally:
        kcp.set_backend("fake", previous)
    assert seen["fwd_prep_l2n"] == ["kn", "qn"]
    if fused_o_tape and kcp.fused_o_tape_shipped():
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "fused_tape"
        assert seen["kcp_fwdh_out_tape"] == ["kn", "qn"]
        assert not {"kcp_fwdh_tape", "kcp_fwdh_out", "fwd_o"} & set(seen)
    else:
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
        assert seen["kcp_fwdh_tape"] == ["kn", "qn"]
        assert seen["fwd_o"] == ["qn"]
        assert not {"kcp_fwdh_out", "kcp_fwdh_out_tape"} & set(seen)
    assert not {"qn", "kn"} & set(ws._tapes)


def test_saved_count_and_the_recompute_output_skip() -> None:
    """``_saved_count`` counts ``qk_rstd`` (``5 + len(saved.tensors())``), so the checkpoint frame proof still holds:
    the recompute of a region the node closes skips #5 with the norm on."""
    ctx, ops = _setup()
    opts = kcp.make_options(backend="fake", use_qk_l2norm_in_kernel=True)
    call, _ = kcp._prepare(
        *_ops(ops), ctx, scale=None, opts=opts, workspace=None, backward=False
    )
    out, saved = kcp._fwd_body(call, save=True)
    assert kcp._saved_count(call, True) == kcp._SAVED_OPERANDS + len(saved.tensors())
    if not kcp.torch_release_verified():
        pytest.skip("the frame proof is verified on another torch release")
    before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
    _autograd(ctx, ops, ckpt=True, use_qk_l2norm_in_kernel=True)
    assert kcp.RECOMPUTE_OUTPUT_STATS["skipped"] - before["skipped"] == 1


# ----------------------------------------------------------------------------------------------------------------
# saved set v2, handle key, options and signatures


def _same_field(a: object, b: object) -> bool:
    """``True`` when ``a`` and ``b`` are the same tensor object, tuples of such, or equal non-tensor values."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(
            _same_field(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return a is b
    return a is b or a == b


def test_saved_set_v2_round_trip_and_refusals() -> None:
    ctx, ops = _setup()
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
        *_ops(ops),
        ctx,
        backend="fake",
        use_qk_l2norm_in_kernel=True,
    )
    assert kcp.KCP_SAVED_VERSION == 2 and saved.version == 2
    assert saved.qk_l2norm is True and saved.flags() == (True, False, True)
    assert saved.tensors()[-1] is saved.qk_rstd and saved.meta()["qk_l2norm"] is True
    back = kcp.CakeGDNNondetKcpSaved.from_tensors(
        saved.tensors(), saved.flags(), saved.meta(), ctx
    )
    # Field by field: on Python >= 3.13 a dataclass ``__eq__`` compares tensor fields with ``==`` (ambiguous).
    for f in dataclasses.fields(saved):
        if f.compare:
            assert _same_field(getattr(back, f.name), getattr(saved, f.name)), f.name
    assert back.qk_rstd is saved.qk_rstd
    bwd = dict(backend="fake")
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        kcp.cake_gdn_nondet_train_kcp_backward(
            *_ops(ops), ctx, ops["dout"], saved=saved, **bwd
        )  # flag off, saved on
    _o, saved_off = kcp.cake_gdn_nondet_train_kcp_forward(
        *_ops(ops), ctx, backend="fake"
    )
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        kcp.cake_gdn_nondet_train_kcp_backward(
            *_ops(ops),
            ctx,
            ops["dout"],
            saved=saved_off,
            use_qk_l2norm_in_kernel=True,
            **bwd,
        )
    no_rstd = dataclasses.replace(saved, qk_rstd=None)
    with pytest.raises(ValueError, match="qk_rstd"):
        kcp.cake_gdn_nondet_train_kcp_backward(
            *_ops(ops),
            ctx,
            ops["dout"],
            saved=no_rstd,
            use_qk_l2norm_in_kernel=True,
            **bwd,
        )
    bad_shape = dataclasses.replace(saved, qk_rstd=saved.qk_rstd[:, :-1])
    with pytest.raises(ValueError, match="qk_rstd"):
        kcp.cake_gdn_nondet_train_kcp_backward(
            *_ops(ops),
            ctx,
            ops["dout"],
            saved=bad_shape,
            use_qk_l2norm_in_kernel=True,
            **bwd,
        )
    v1 = dataclasses.replace(saved, version=1)
    with pytest.raises(ValueError, match="version"):
        kcp.cake_gdn_nondet_train_kcp_backward(
            *_ops(ops),
            ctx,
            ops["dout"],
            saved=v1,
            use_qk_l2norm_in_kernel=True,
            **bwd,
        )
    owned = dataclasses.replace(saved, release_in_backward=True)
    owned.forget(kcp.QK_RSTD)
    assert owned.qk_rstd is None and saved.qk_rstd is not None


def test_handle_key_carries_the_flag() -> None:
    """A handle entry of the other form is refused (``S_in`` is a function of ``kn``): the key has the flag."""
    ctx, ops = _setup()
    args = (1000, 2, 6, 4, 1, 0.1, "a2a", 0, "fla", "tf32_hilo")
    assert kcp._handle_key(*args, True) != kcp._handle_key(*args, False)
    opts = kcp.make_options(
        backend="fake", use_qk_l2norm_in_kernel=True, save_transfer=False
    )
    _out, saved = kcp.cake_gdn_nondet_train_kcp_forward(*_ops(ops), ctx, options=opts)
    key_off = kcp._handle_key(
        saved.T_r,
        saved.H_qk,
        saved.HV,
        saved.n_frag,
        saved.P,
        saved.scale,
        saved.exchange,
        saved.precision_mode,
        saved.offset_class,
        saved.transfer_walk,
        False,
    )
    with pytest.raises(ValueError, match="another problem"):
        kcp._fwd_entry(
            *_ops(ops),
            ctx,
            scale=None,
            save_for_backward=True,
            workspace=None,
            opts=opts,
            hand_off=False,
            boundary={"ctx": ctx, "key": key_off, "S_in": saved.S_in},
        )


def test_option_keyword_and_override_rule() -> None:
    """One public keyword: a ``CakeGDNNondetKcpOptions`` field (bool only), keyword-only on the three entry points,
    following the literal-override rule (an ``options`` instance wins over the default keyword); no rstd keyword (the
    facade's hand-off: ``test_cake_gdn_nondet_train_kcp_api.py``)."""
    opt = _mod("cake_options")
    assert opt.make_options().use_qk_l2norm_in_kernel is False
    assert "use_qk_l2norm_in_kernel" in opt.OPTION_NAMES
    for bad in (1, "yes", None):
        with pytest.raises(TypeError):
            opt.make_options(use_qk_l2norm_in_kernel=bad)
    assert (
        opt.make_options(use_qk_l2norm_in_kernel=True).key() != opt.make_options().key()
    )  # the graph key
    for fn in (
        kcp.cake_gdn_nondet_kcp,
        kcp.cake_gdn_nondet_train_kcp_forward,
        kcp.cake_gdn_nondet_train_kcp_backward,
    ):
        params = inspect.signature(fn).parameters
        p = params["use_qk_l2norm_in_kernel"]
        assert p.kind == p.KEYWORD_ONLY and p.default is False, fn
        assert not any("rstd" in n for n in params), fn
    ctx, ops = _setup()
    on = kcp.make_options(backend="fake", use_qk_l2norm_in_kernel=True)
    kcp.cake_gdn_nondet_train_kcp_forward(*_ops(ops), ctx, options=on)
    assert (
        kcp.LAST_CALL["fwd"]["qk_l2norm"] is True
    )  # the instance's value: the default keyword does not override
    kcp.cake_gdn_nondet_train_kcp_forward(
        *_ops(ops),
        ctx,
        options=kcp.make_options(backend="fake"),
        use_qk_l2norm_in_kernel=True,
    )
    assert kcp.LAST_CALL["fwd"]["qk_l2norm"] is True  # a non-default keyword overrides
    kcp.cake_gdn_nondet_train_kcp_forward(
        *_ops(ops),
        ctx,
        options={"backend": "fake", "use_qk_l2norm_in_kernel": True},
    )
    assert kcp.LAST_CALL["fwd"]["qk_l2norm"] is True
    leaves = [x.detach().clone().requires_grad_(True) for x in _ops(ops)]
    kcp.cake_gdn_nondet_kcp(
        *leaves, ctx, backend="fake", use_qk_l2norm_in_kernel=True
    ).backward(ops["dout"])
    assert (
        kcp.LAST_CALL["fwd"]["qk_l2norm"] is True
        and kcp.LAST_CALL["bwd"]["qk_l2norm"] is True
    )


def test_kernel_backend_refuses_until_the_records_ship() -> None:
    """Without the three records in the registry the kernel backend refuses the option before any launch, with a
    clear message; the fake backend runs it."""
    ctx, ops = _setup()
    assert kcp.l2n_records_shipped() == jit.l2n_records_shipped()
    if kcp.l2n_records_shipped():
        pytest.skip("the registry ships the in-kernel L2 norm records")
    with pytest.raises(RuntimeError, match="not ship"):
        kcp.cake_gdn_nondet_train_kcp_forward(
            *_ops(ops),
            ctx,
            backend="kernels",
            use_qk_l2norm_in_kernel=True,
        )
    with pytest.raises(ValueError, match="no shipped record"):
        jit.select_record("fwd_prep_l2n")
    assert jit.records_of_program("qk_apply") == ()


def test_launch_values_are_the_program_args() -> None:
    """Every launch of the three programs passes exactly ``PROGRAM_ARGS[program]`` (logical names of the registry
    vocabulary); the siblings' extras are ``PROGRAM_ARGS_L2N``; the planned grids are the plan's rule."""
    ctx, ops = _setup()
    seen: dict[str, tuple[set, tuple]] = {}

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            record, values, grid, _t = kcp.resolve_launch(name, args)
            seen[name] = (set(values), grid)
            super().run(name, args)

    previous = kcp.set_backend("fake", Recording())
    try:
        _saved_pass(ctx, ops, use_qk_l2norm_in_kernel=True, save_mode="full")
        _saved_pass(ctx, ops, use_qk_l2norm_in_kernel=True)
    finally:
        kcp.set_backend("fake", previous)
    vocab = set(jit.LOGICAL_TENSORS) | set(jit.LOGICAL_SCALARS)
    for prog in kcp.L2N_PROGRAMS:
        names, grid = seen[prog]
        want = set(kcp.PROGRAM_ARGS[prog])
        # fwd_prep's launches carry num_seqs beyond its table, as the v1 #1's do
        assert want <= names and names - want <= {"num_seqs"}, (
            prog,
            sorted(names ^ want),
        )
        assert want - set(kcp.PROVISIONAL_ARGS) <= vocab, prog
    for v1, prog in kcp.L2N_SIBLING_OF.items():
        assert set(kcp.PROGRAM_ARGS[prog]) == set(kcp.PROGRAM_ARGS[v1]) | set(
            kcp.PROGRAM_ARGS_L2N[prog]
        )
    NC = int(ctx.NC)
    assert seen["fwd_prep_l2n"][1] == (NC, 2, 1) and seen["qk_apply"][1] == (NC, 2, 1)
    assert set(jit.L2N_PROGRAMS) == set(kcp.L2N_PROGRAMS) and set(
        jit.L2N_RECORD_PLANS
    ) == set(kcp.L2N_PROGRAMS)
    assert jit.L2N_RECORD_PLANS["fwd_prep_l2n"]["resident_ctas_per_sm"] == 2
    assert kcp.TART_MODULE_OF["fwd_prep_l2n"] == "gdn_chunk_det_train_fwd_prep_l2n_v1"


class _PeakWorkspace(kcp.CakeGDNNondetKcpWorkspace):
    """A workspace recording the most bytes it binds (every allocation path)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.peak = 0

    def _upd(self):
        self.peak = max(self.peak, self.bytes)

    def tape(self, name):
        t = super().tape(name)
        self._upd()
        return t

    def pool(self, name, n=None):
        t = super().pool(name, n)
        self._upd()
        return t

    def ensure(self, *a, **k):
        tb = super().ensure(*a, **k)
        self._upd()
        return tb

    def adopt_tape(self, name, t):
        r = super().adopt_tape(name, t)
        self._upd()
        return r


def test_kept_backward_binds_the_recompute_rows() -> None:
    """A checkpoint with kept recompute tapes and the norm: the recompute forward hands its ``qn`` / ``kn`` over with
    the tapes, the backward binds them (``qk_source`` ``"kept"``) and launches neither ``qk_apply`` nor #1_l2n; the
    rows it reads are the forward's (``record_qk``), the gradients the direct autograd's, nothing of the norm and no
    kept entry is left after the backward (#7039), and the arena follows the kept form of the workspace query."""
    from torch.utils.checkpoint import checkpoint

    ctx, ops = _setup()
    H_qk, HV = ops["q"].shape[1], ops["v"].shape[1]
    o = dict(use_qk_l2norm_in_kernel=True)
    direct = _autograd(ctx, ops, ckpt=False, **o)
    # a checkpointed call saves as "light" (the default "light_akk" too): it is sized with save_mode="light"
    layout = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
        ctx, H_qk=H_qk, HV=HV, save_mode="light", reuse_recompute_tapes=True, **o
    )
    ws = _PeakWorkspace()
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def fn(q, k, v, g, beta):
        return kcp.cake_gdn_nondet_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            workspace=ws,
            backend="fake",
            reuse_recompute_tapes=True,
            test_knobs={"record_qk": True},
            **o,
        )

    stats = dict(kcp.KEPT_STATS)
    out = checkpoint(fn, *leaves, use_reentrant=False)
    assert ws.peak == layout["forward"], (ws.peak, layout["forward"])
    ws.peak = 0
    out.backward(ops["dout"])
    assert ws.peak == layout["backward"], (ws.peak, layout["backward"])
    assert ws.bytes == layout["bwd.phase.end"], (ws.bytes, layout["bwd.phase.end"])
    rec = kcp.LAST_CALL["bwd"]
    assert rec["tapes_source"] == "kept" and rec["qk_source"] == "kept"
    assert rec["rstd_source"] == "saved"
    assert not {"qk_apply", "fwd_prep_l2n"} & set(rec["sequence"]), rec["sequence"]
    assert "bwd_dqkg_l2n" in rec["sequence"]
    for key in ("take", "released"):
        assert kcp.KEPT_STATS[key] - stats[key] == 1, key
    assert kcp.kept_registry_size() == 0
    qh, rq = l2n.qk_l2norm_ref(ops["q"])
    kh, rk = l2n.qk_l2norm_ref(ops["k"])
    assert torch.equal(rec["qk"]["qn"], qh) and torch.equal(rec["qk"]["kn"], kh)
    assert torch.equal(rec["qk"]["rstd"], torch.stack([rq, rk]))
    assert not set(ws._tapes) & {*kcp.QK_TAPES, kcp.QK_RSTD}, sorted(ws._tapes)
    got = {
        "out": out.detach(),
        **{n: lf.grad for n, lf in zip(GRADS, leaves, strict=True)},
    }
    _same(direct, got, "kept checkpoint")


def test_kept_rows_are_checked() -> None:
    """A hand-off forward unbinds ``qn`` / ``kn`` from the arena and hands them over exactly when the norm is on;
    a kept set without them (or with them while the call runs without the norm) is refused by the backward."""
    ctx, ops = _setup()
    for on in (True, False):
        # the hand-off is the checkpointed form, which saves as "light" (with the norm
        # too, where an unset save_mode is "light_akk")
        o = dict(
            backend="fake",
            use_qk_l2norm_in_kernel=on,
            save_mode="light",
            reuse_recompute_tapes=True,
        )
        ws = kcp.CakeGDNNondetKcpWorkspace()
        call, _ = kcp._prepare(
            *_ops(ops),
            ctx,
            scale=None,
            opts=kcp.make_options(**o),
            workspace=ws,
            backward=False,
        )
        _out, saved = kcp._fwd_body(call, save=True, hand_off=True)
        assert set(saved.kept.qk) == (set(kcp.QK_TAPES) if on else set())
        assert not set(ws._tapes) & {*kcp.QK_TAPES, kcp.QK_RSTD}, sorted(ws._tapes)
        saved.kept.qk = (
            {}
            if on
            else {
                n: torch.zeros(ops["q"].shape, dtype=torch.bfloat16)
                for n in kcp.QK_TAPES
            }
        )
        with pytest.raises(ValueError, match="qn / kn"):
            kcp.cake_gdn_nondet_train_kcp_backward(
                *_ops(ops),
                ctx,
                ops["dout"],
                saved=saved,
                workspace=ws,
                **o,
            )


def test_workspace_layout_counts_the_norm_rows() -> None:
    """The workspace query: ``qn`` / ``kn`` in the forward and every backward phase (the kept-tape form: hand-off
    tensors, ``bwd.kept_tapes``, outside the workspace), the arena ``r`` only where #1 is re-run; flag off: the v1
    layout exactly."""
    ctx, ops = _setup()
    H_qk, HV = ops["q"].shape[1], ops["v"].shape[1]
    T = ops["q"].shape[0]
    row = T * H_qk * 128 * 2
    for kw in (
        {},
        {"save_mode": "full"},
        {"boundary_mode": "replay"},
        {"reuse_recompute_tapes": True},
    ):
        off = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, **kw
        )
        on = gbytes.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx,
            H_qk=H_qk,
            HV=HV,
            use_qk_l2norm_in_kernel=True,
            # the norm's terms against the same save mode (the default with the norm is "light_akk")
            **{"save_mode": "light", **kw},
        )
        kept = bool(kw.get("reuse_recompute_tapes"))
        assert on["fwd.qn"] == on["fwd.kn"] == row
        assert on["bwd.qn"] == on["bwd.kn"] == (0 if kept else row), kw
        assert on["bwd.kept_tapes"] - off["bwd.kept_tapes"] == (2 * row if kept else 0)
        assert on["forward"] - off["forward"] == 2 * row
        replays = kw.get("boundary_mode") == "replay" or not (
            kw.get("save_mode") == "full" or kept
        )
        assert on["bwd.qk_rstd"] == (8 * T * H_qk if replays else 0), kw
        for ph in ("exchange", "walk", "end"):
            assert (
                on[f"bwd.phase.{ph}"] - off[f"bwd.phase.{ph}"]
                == (0 if kept else 2 * row) + on["bwd.qk_rstd"]
            ), (kw, ph)
        if kept:
            assert (
                on["bwd.phase.recompute_forward"] - off["bwd.phase.recompute_forward"]
                == 2 * row
            )
        assert (
            on["backward_transient"] - off["backward_transient"]
            == 2 * row + on["bwd.qk_rstd"]
        ), kw
        assert not any(k.endswith((".qn", ".kn", ".qk_rstd")) for k in off)
    with pytest.raises(TypeError):
        gbytes.cake_gdn_nondet_train_kcp_workspace_bytes(
            ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=1
        )


def test_emulation_threads_the_flag() -> None:
    """The W-rank emulation with the norm: saved == replay, and ``out`` / ``dv`` / ``dbeta`` / ``dg`` are the norm-off
    emulation's on ``(q_hat, k_hat)`` (CP2, one value head per Q/K head: the fused sum is the partial)."""
    cu = [0, 300, 640, 900]
    H_qk = HV = 2
    gl = _raw_inputs(cu, H_qk, HV, seed=11)
    part = plan.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 2, "contiguous"
    )
    on = fake.emulate(part, gl, H_qk=H_qk, HV=HV, qk_l2norm=True)
    assert fake.replay_equals_saved(on) and on["qk_l2norm"] is True
    normed = dict(gl)
    normed["q"], _ = l2n.qk_l2norm_ref(gl["q"])
    normed["k"], _ = l2n.qk_l2norm_ref(gl["k"])
    off = fake.emulate(part, normed, H_qk=H_qk, HV=HV)
    for n in ("out", "dv", "dbeta", "dg"):
        assert torch.equal(on["saved"][n], off["saved"][n]), n
    assert not torch.equal(on["saved"]["dq"], off["saved"]["dq"])


def test_zero_token_rank_saves_an_empty_rstd() -> None:
    """``T_r = 0``: nothing launches, the saved set still carries ``qk_rstd`` (``[2, 0, H_qk]``) so the flags agree
    across ranks, and the backward accepts it."""
    try:
        part = plan.cake_gdn_nondet_train_kcp_partition(
            torch.tensor([0, 0], dtype=torch.int64), 1, "contiguous"
        )
        ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    except Exception as exc:  # noqa: BLE001 - a partition without tokens may be refused at build
        pytest.skip(f"no zero-token single-rank context: {exc}")
    ops = _raw_inputs([0, 0], 2, 6)
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
        *_ops(ops),
        ctx,
        backend="fake",
        use_qk_l2norm_in_kernel=True,
    )
    assert (
        tuple(saved.qk_rstd.shape) == (2, 0, 2) and saved.tensors()[-1] is saved.qk_rstd
    )
    grads = kcp.cake_gdn_nondet_train_kcp_backward(
        *_ops(ops),
        ctx,
        ops["dout"],
        saved=saved,
        backend="fake",
        use_qk_l2norm_in_kernel=True,
    )
    assert all(t.shape[0] == 0 for t in grads)


def test_no_fla_import() -> None:
    ctx, ops = _setup()
    _saved_pass(ctx, ops, use_qk_l2norm_in_kernel=True)
    assert "fla" not in sys.modules


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float(
        ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()
    )


@pytest.mark.parametrize("heads", [(2, 2), (2, 6)])
def test_det_package_with_the_norm_within_the_det_constants(heads) -> None:
    """The deterministic package with the norm on the same raw inputs and context (fake backend): every tensor within
    the deterministic accuracy constants of this package (rel-RMS <= 1e-2, max-abs <= 2e-2 of max|ref|), worst over
    both save modes and the checkpoint replay."""
    det_kcp = _det_mod("cake_kcp")
    det_plan = _det_mod("cake_kcp_plan")
    H_qk, HV = heads
    ops = _raw_inputs(CU, H_qk, HV)
    part = plan.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(CU, dtype=torch.int64), 1, "contiguous"
    )
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    det_part = det_plan.cake_gdn_det_train_kcp_partition(
        torch.tensor(CU, dtype=torch.int64), 1, "contiguous"
    )
    det_ctx = det_plan.cake_gdn_det_train_kcp_build_context(det_part, None, 0, CPU)
    worst = {}
    for save_mode in ("light", "full"):
        nd = _saved_pass(ctx, ops, use_qk_l2norm_in_kernel=True, save_mode=save_mode)
        o = dict(backend="fake", use_qk_l2norm_in_kernel=True, save_mode=save_mode)
        out, saved = det_kcp.cake_gdn_det_train_kcp_forward(*_ops(ops), det_ctx, **o)
        dq, dk, dv, dbeta, dg = det_kcp.cake_gdn_det_train_kcp_backward(
            *_ops(ops), det_ctx, ops["dout"], saved=saved, **o
        )
        det = {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}
        for n in TENSORS:
            ref = det[n].double()
            rr = _rel_rms(nd[n], det[n])
            ma = float((nd[n].double() - ref).abs().max().item())
            bound = 2e-2 * float(ref.abs().max().item())
            worst[n] = max(worst.get(n, 0.0), rr)
            assert rr <= 1e-2 and ma <= bound + 1e-30, (
                heads,
                save_mode,
                n,
                rr,
                ma,
                bound,
            )
    assert set(worst) == set(TENSORS)
