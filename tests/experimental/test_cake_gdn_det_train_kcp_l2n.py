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

CPU tests of the in-kernel Q / K L2 norm of the KCP GDN route (``use_qk_l2norm_in_kernel``):

* the pure-torch reference ``cake_l2n`` against an exact rational oracle (every FP32 step rounded from the exact
  value; subnormals, zero rows, overflow of the sum of squares, wide exponents), in the documented orders;
* the fake backend's three programs (``fwd_prep_l2n`` / ``bwd_dqkg_l2n`` / ``qk_apply``) against the reference and
  against the norm-off programs on the normalized rows;
* the host plumbing on the fake backend: the composed chain (helper -> norm-off route on ``(q_hat, k_hat)`` ->
  helper pullback) bitwise, every saved / replay / checkpoint / handle form bitwise the direct one, ``qn`` / ``kn``
  never retained, the saved set v2, the option and the facade keyword, the byte queries.

The fake backend is not bitwise the device programs in #1 / #8's bodies (different accumulation orders): these tests
prove the plumbing and the oracle, not GPU numerics (``test_cake_gdn_det_train_kcp_dist.py`` holds the GPU rows).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
import struct
import sys
from fractions import Fraction

import pytest
import torch

PACKAGE = "flashinfer.experimental.cake_gdn_det_train_kcp"


def _mod(sub: str | None = None):
    try:
        from tests.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_det_train_kcp", sub)
    except (ImportError, KeyError):
        return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


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
    part = plan.partition_det_kcp(torch.tensor(cu, dtype=torch.int64), 1, "contiguous")
    ctx = plan.build_context_det_kcp(part, None, 0, CPU)
    return ctx, _raw_inputs(cu, H_qk, HV)


def _ops(ops):
    return tuple(ops[n] for n in ("q", "k", "v", "g", "beta"))


def _saved_pass(ctx, ops, *, q=None, k=None, ws=None, **o) -> dict[str, torch.Tensor]:
    qq = ops["q"] if q is None else q
    kk = ops["k"] if k is None else k
    o = {"backend": "fake", **o}
    out, saved = kcp.chunk_gdn_det_fwd_kcp(
        qq, kk, ops["v"], ops["g"], ops["beta"], ctx, workspace=ws, **o
    )
    dq, dk, dv, dbeta, dg = kcp.chunk_gdn_det_bwd_kcp(
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
    hd = kcp.RecomputeHandleDetKcp() if handle else None

    def fn(q, k, v, g, beta, h):
        return kcp.chunk_gdn_det_kcp(q, k, v, g, beta, ctx, recompute_handle=h, **o)

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
        workspace=kcp.WorkspaceDetKcp(),
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
    if group_reduce == "fused" and HV != H_qk:
        pytest.skip(
            "the fused FP32 group sum of the emulation is not the head-order sum of its partials"
        )
    ctx, ops = _setup(H_qk, HV)
    on = _saved_pass(ctx, ops, use_qk_l2norm_in_kernel=True, group_reduce=group_reduce)
    qh, rq = l2n.qk_l2norm_ref(ops["q"])
    kh, rk = l2n.qk_l2norm_ref(ops["k"])
    ws = kcp.WorkspaceDetKcp()
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
            assert kcp.LAST_CALL["bwd"]["qk_source"] == "qk_apply"
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
    pq, pk, pv = kcp.packed_qkv_views_det_kcp(qkv, H_qk=H, HV=HV)
    out = kcp.chunk_gdn_det_kcp(pq, pk, pv, g, beta, ctx, backend="fake", **o)
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


def test_norm_off_is_the_v1_route() -> None:
    """With the flag off nothing of the norm runs or is bound: the v1 program sequence, no qn / kn / rstd tape,
    the v1 saved set (no ``qk_rstd``), and ``LAST_CALL`` says so."""
    ctx, ops = _setup()
    ws = kcp.WorkspaceDetKcp()
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
    out, saved = kcp.chunk_gdn_det_fwd_kcp(*_ops(ops), ctx, backend="fake", **o)
    fwd = kcp.LAST_CALL["fwd"]["qk"]
    kcp.chunk_gdn_det_bwd_kcp(
        *_ops(ops), ctx, ops["dout"], saved=saved, backend="fake", **o
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
        {},
        {"save_mode": "full"},
        {"boundary_mode": "replay"},
        {"save_for_backward": False},
    ):
        ws = kcp.WorkspaceDetKcp()
        out, saved = kcp.chunk_gdn_det_fwd_kcp(
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
        _o, saved_off = kcp.chunk_gdn_det_fwd_kcp(*_ops(ops), ctx, backend="fake", **opts)
        assert saved.bytes - saved_off.bytes == 8 * T * H_qk
        terms = gbytes.gdn_det_saved_layout_kcp(
            ctx,
            H_qk=H_qk,
            HV=HV,
            use_qk_l2norm_in_kernel=True,
            **{k: v for k, v in opts.items()},
        )
        assert terms["qk_rstd"] == 8 * T * H_qk
        assert (
            gbytes.gdn_det_saved_bytes_kcp(
                ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=True, **opts
            )
            - (gbytes.gdn_det_saved_bytes_kcp(ctx, H_qk=H_qk, HV=HV, **opts))
            == 8 * T * H_qk
        )
        kcp.chunk_gdn_det_bwd_kcp(
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
    assert "qk_rstd" not in gbytes.gdn_det_saved_layout_kcp(
        ctx, H_qk=H_qk, HV=HV
    )  # norm-off layout unchanged


def test_forward_drops_kn_before_the_output_and_qn_after_fwd_o() -> None:
    """The arena at #5 holds ``qn`` but no ``kn`` (the tape walks were its last readers); ``qn`` goes after #5."""
    ctx, ops = _setup()
    seen = {}

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if not call.backward:
                seen[name] = sorted(set(call.ws._tapes) & {"qn", "kn", kcp.QK_RSTD})
            super().run(name, args)

    previous = kcp.set_backend("fake", Recording())
    try:
        kcp.chunk_gdn_det_fwd_kcp(
            *_ops(ops),
            ctx,
            backend="fake",
            workspace=kcp.WorkspaceDetKcp(),
            use_qk_l2norm_in_kernel=True,
        )
    finally:
        kcp.set_backend("fake", previous)
    assert seen["fwd_prep_l2n"] == [
        "kn",
        "qn",
    ]  # the saved r is a fresh tensor, never the arena's
    assert seen["kcp_fwdh_tape"] == ["kn", "qn"]
    assert seen["fwd_o"] == ["qn"]


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


def test_saved_set_v2_round_trip_and_refusals() -> None:
    ctx, ops = _setup()
    out, saved = kcp.chunk_gdn_det_fwd_kcp(
        *_ops(ops), ctx, backend="fake", use_qk_l2norm_in_kernel=True
    )
    assert kcp.KCP_SAVED_VERSION == 2 and saved.version == 2
    assert saved.qk_l2norm is True and saved.flags() == (True, False, True)
    assert saved.tensors()[-1] is saved.qk_rstd and saved.meta()["qk_l2norm"] is True
    back = kcp.SavedDetKcp.from_tensors(saved.tensors(), saved.flags(), saved.meta(), ctx)
    assert back == saved and back.qk_rstd is saved.qk_rstd
    bwd = dict(backend="fake")
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        kcp.chunk_gdn_det_bwd_kcp(
            *_ops(ops), ctx, ops["dout"], saved=saved, **bwd
        )  # flag off, saved on
    _o, saved_off = kcp.chunk_gdn_det_fwd_kcp(*_ops(ops), ctx, backend="fake")
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        kcp.chunk_gdn_det_bwd_kcp(
            *_ops(ops),
            ctx,
            ops["dout"],
            saved=saved_off,
            use_qk_l2norm_in_kernel=True,
            **bwd,
        )
    no_rstd = dataclasses.replace(saved, qk_rstd=None)
    with pytest.raises(ValueError, match="qk_rstd"):
        kcp.chunk_gdn_det_bwd_kcp(
            *_ops(ops),
            ctx,
            ops["dout"],
            saved=no_rstd,
            use_qk_l2norm_in_kernel=True,
            **bwd,
        )
    bad_shape = dataclasses.replace(saved, qk_rstd=saved.qk_rstd[:, :-1])
    with pytest.raises(ValueError, match="qk_rstd"):
        kcp.chunk_gdn_det_bwd_kcp(
            *_ops(ops),
            ctx,
            ops["dout"],
            saved=bad_shape,
            use_qk_l2norm_in_kernel=True,
            **bwd,
        )
    v1 = dataclasses.replace(saved, version=1)
    with pytest.raises(ValueError, match="version"):
        kcp.chunk_gdn_det_bwd_kcp(
            *_ops(ops), ctx, ops["dout"], saved=v1, use_qk_l2norm_in_kernel=True, **bwd
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
    _out, saved = kcp.chunk_gdn_det_fwd_kcp(*_ops(ops), ctx, options=opts)
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
    """One public keyword: a ``OptionsDetKcp`` field (bool only), keyword-only on the three entry points and the facade,
    following the literal-override rule (an ``options`` instance wins over the default keyword); no rstd keyword."""
    opt = _mod("cake_options")
    assert opt.make_options().use_qk_l2norm_in_kernel is False
    assert "use_qk_l2norm_in_kernel" in opt.OPTION_NAMES
    for bad in (1, "yes", None):
        with pytest.raises(TypeError):
            opt.make_options(use_qk_l2norm_in_kernel=bad)
    assert (
        opt.make_options(use_qk_l2norm_in_kernel=True).key() != opt.make_options().key()
    )  # the graph key
    facade = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")
    for fn in (
        kcp.chunk_gdn_det_kcp,
        kcp.chunk_gdn_det_fwd_kcp,
        kcp.chunk_gdn_det_bwd_kcp,
        facade.chunk_gdn_det_kcp,
        facade.chunk_gdn_det_fwd_kcp,
        facade.chunk_gdn_det_bwd_kcp,
    ):
        params = inspect.signature(fn).parameters
        p = params["use_qk_l2norm_in_kernel"]
        assert p.kind == p.KEYWORD_ONLY and p.default is False, fn
        assert not any("rstd" in n for n in params), fn
    ctx, ops = _setup()
    on = kcp.make_options(backend="fake", use_qk_l2norm_in_kernel=True)
    kcp.chunk_gdn_det_fwd_kcp(*_ops(ops), ctx, options=on)
    assert (
        kcp.LAST_CALL["fwd"]["qk_l2norm"] is True
    )  # the instance's value: the default keyword does not override
    kcp.chunk_gdn_det_fwd_kcp(
        *_ops(ops),
        ctx,
        options=kcp.make_options(backend="fake"),
        use_qk_l2norm_in_kernel=True,
    )
    assert kcp.LAST_CALL["fwd"]["qk_l2norm"] is True  # a non-default keyword overrides
    kcp.chunk_gdn_det_fwd_kcp(
        *_ops(ops), ctx, options={"backend": "fake", "use_qk_l2norm_in_kernel": True}
    )
    assert kcp.LAST_CALL["fwd"]["qk_l2norm"] is True
    leaves = [x.detach().clone().requires_grad_(True) for x in _ops(ops)]
    facade.chunk_gdn_det_kcp(
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
        kcp.chunk_gdn_det_fwd_kcp(
            *_ops(ops), ctx, backend="kernels", use_qk_l2norm_in_kernel=True
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


def test_workspace_layout_counts_the_norm_rows() -> None:
    """The workspace query: ``qn`` / ``kn`` in the forward and every backward phase, the arena ``r`` only where #1
    is re-run; flag off: the v1 layout exactly; the measured arena of a light backward fits it."""
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
        off = gbytes.gdn_det_workspace_layout_kcp(ctx, H_qk=H_qk, HV=HV, **kw)
        on = gbytes.gdn_det_workspace_layout_kcp(
            ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=True, **kw
        )
        assert on["fwd.qn"] == on["fwd.kn"] == on["bwd.qn"] == on["bwd.kn"] == row
        assert on["forward"] - off["forward"] == 2 * row
        replays = kw.get("boundary_mode") == "replay" or not (
            kw.get("save_mode") == "full" or kw.get("reuse_recompute_tapes")
        )
        assert on["bwd.qk_rstd"] == (8 * T * H_qk if replays else 0), kw
        for ph in ("exchange", "walk", "end"):
            assert (
                on[f"bwd.phase.{ph}"] - off[f"bwd.phase.{ph}"]
                == 2 * row + on["bwd.qk_rstd"]
            ), (kw, ph)
        assert not any(k.endswith((".qn", ".kn", ".qk_rstd")) for k in off)
    with pytest.raises(TypeError):
        gbytes.gdn_det_workspace_bytes_kcp(ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=1)


def test_emulation_threads_the_flag() -> None:
    """The W-rank emulation with the norm: saved == replay, and ``out`` / ``dv`` / ``dbeta`` / ``dg`` are the norm-off
    emulation's on ``(q_hat, k_hat)`` (CP2, one value head per Q/K head: the fused sum is the partial)."""
    cu = [0, 300, 640, 900]
    H_qk = HV = 2
    gl = _raw_inputs(cu, H_qk, HV, seed=11)
    part = plan.partition_det_kcp(torch.tensor(cu, dtype=torch.int64), 2, "contiguous")
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
        part = plan.partition_det_kcp(
            torch.tensor([0, 0], dtype=torch.int64), 1, "contiguous"
        )
        ctx = plan.build_context_det_kcp(part, None, 0, CPU)
    except Exception as exc:  # noqa: BLE001 - a partition without tokens may be refused at build
        pytest.skip(f"no zero-token single-rank context: {exc}")
    ops = _raw_inputs([0, 0], 2, 6)
    out, saved = kcp.chunk_gdn_det_fwd_kcp(
        *_ops(ops), ctx, backend="fake", use_qk_l2norm_in_kernel=True
    )
    assert (
        tuple(saved.qk_rstd.shape) == (2, 0, 2) and saved.tensors()[-1] is saved.qk_rstd
    )
    grads = kcp.chunk_gdn_det_bwd_kcp(
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
