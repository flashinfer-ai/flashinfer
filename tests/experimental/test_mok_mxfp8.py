# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU native MXFP8 MoK checks against a quantization-aware reference.

The reference follows the MoK recipe in schedule order: x, the SwiGLU
activation, the score-scaled dY and d_gate/d_up are MXFP8 (32-element blocks
along K), routed weights are caller-prequantized, GEMMs accumulate in FP32 on
dequantized operands and round to BF16, and the shared expert stays BF16.

The routed gate, up and dh are BF16 roundings of FP32 sums that the kernel
accumulates in another order. Where such a sum lies within the accumulation
band of a BF16 rounding midpoint (``_tie_candidates``), the kernel may round
to the other BF16 neighbour; the training reference evaluates that neighbour
too and adds the exact effect of the E4M3 code flips it causes to the
tolerance of the gradient elements they reach. The ``1e-2 + 1e-2 * |reference|``
gate itself is unchanged, and no allowance exists where no input is ambiguous.

Backend suite: the package is imported only through
:func:`tests.test_helpers.cake_package.cake_package`; the :mod:`flashinfer.mok`
wrapper is covered by ``test_mok_bf16_api.py``.
"""

import pytest
import torch

from tests.experimental._mok_reference import (
    mxfp8_dequantize as dequantize,
    mxfp8_quantize as quantize,
    require_gpu,
    single_rank_schedule,
    swiglu,
)
from tests.test_helpers.cake_package import cake_package

kernels = cake_package("cake_mok_bf16", "_kernels")
api = cake_package("cake_mok_bf16", "api")
MoKBackwardMXFP8 = kernels.MoKBackwardMXFP8
MoKForwardMXFP8 = kernels.MoKForwardMXFP8
MoKRecomputeMXFP8 = kernels.MoKRecomputeMXFP8
quantize_mok_mxfp8_weights = api.quantize_mok_mxfp8_weights

LIMITS = pytest.mark.parametrize("swiglu_limit", [None, 0.75], ids=["plain", "clamped"])


@pytest.fixture(autouse=True)
def _exact_matmul(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def _setup(hidden, intermediate, seed, experts=4, tokens=512, topk=2):
    torch.manual_seed(seed)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, **options)
    dy = torch.randn(tokens, hidden, **options) * 0.125
    shared = [
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    scores = torch.rand(tokens, topk, device="cuda") + 0.1
    scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    return x, dy, shared, routed, scores, single_rank_schedule(routes, experts)


def _close(a, b, allowance=None):
    """Elements of ``a`` outside ``1e-2 + 1e-2 * |b|`` (plus the elementwise ``allowance`` when given)."""
    d = (a.float() - b.float()).abs()
    bound = 1e-2 + 1e-2 * b.float().abs()
    if allowance is not None:
        bound = bound + allowance
    return int((d > bound).sum())


def _bf16_toward(value, target):
    """The BF16 neighbour of ``value`` (BF16) on the side of ``target``; ``value`` itself where they are equal."""
    bits = value.view(torch.int16).to(torch.int32) & 0xFFFF
    stepped = torch.where(target.abs() > value.float().abs(), bits + 1, bits - 1)
    from_zero = torch.where(target < 0, 0x8001, 0x0001)
    bits = torch.where(value == 0, from_zero, stepped)
    bits = torch.where(
        target == value.float(), value.view(torch.int16).to(torch.int32) & 0xFFFF, bits
    )
    return (
        torch.where(bits >= 0x8000, bits - 0x10000, bits)
        .to(torch.int16)
        .view(torch.bfloat16)
    )


def _tie_candidates(value, exact, magnitude):
    """Which BF16 GEMM outputs the kernel may round to the other neighbour, and that neighbour.

    ``value`` is the reference's BF16 rounding of an FP32 sum of dequantized
    products, ``exact`` the FP64 value of that sum and ``magnitude`` the sum
    of the absolute products. The kernel accumulates the same products in
    another FP32 order; the two orders differ by a few roundings of a running
    sum bounded by ``magnitude``, each at most 2^-24 of it, so the kernel's
    FP32 result can fall on the other side of a BF16 rounding midpoint only
    when ``exact`` lies within a few units of ``2**-24 * magnitude`` of it.
    The band is one unit: the observed ties sit within a quarter of a unit of
    the midpoint, and one unit keeps the allowance to the elements that are
    ambiguous in practice. Returns the mask of those outputs and ``value``
    with the other neighbour substituted there.
    """
    other = _bf16_toward(value, exact)
    midpoint = (value.float() + other.float()) * 0.5
    band = 2.0**-24 * magnitude
    ambiguous = ((exact - midpoint).abs() <= band) & (other != value)
    return ambiguous, torch.where(ambiguous, other, value)


def test_weight_quantizer_matches_recipe():
    require_gpu()
    torch.manual_seed(17)
    gate = torch.randn(3, 256, 512, device="cuda", dtype=torch.bfloat16)
    gate[0, 5, 64:96] = 0  # an all-zero block takes the minimum scale
    gate[1, 7, :32] *= 1e4  # large blocks saturate to the finite range
    up = torch.randn_like(gate)
    down = torch.randn(3, 512, 256, device="cuda", dtype=torch.bfloat16)
    forward_w, backward_w = quantize_mok_mxfp8_weights(gate, up, down)
    for weight, fwd, bwd in zip((gate, up), forward_w[:2], backward_w[:2], strict=True):
        expected = quantize(weight)
        assert all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(bwd, expected, strict=True)
        )
        assert fwd[0] is bwd[0] and fwd[1] is bwd[1]
    expected_down = quantize(down)
    assert torch.equal(
        forward_w[2][0].view(torch.uint8), expected_down[0].view(torch.uint8)
    )
    assert torch.equal(forward_w[2][1], expected_down[1])
    assert torch.equal(
        backward_w[2][0].view(torch.uint8), expected_down[2].view(torch.uint8)
    )
    assert torch.equal(backward_w[2][1], expected_down[3])
    rel = (dequantize(*expected[:2]) - up.float()).norm() / up.float().norm()
    assert rel.item() < 0.05


@LIMITS
@pytest.mark.parametrize(
    "hidden,intermediate,macro", [(512, 256, 768), (512, 512, 768), (512, 256, 1536)]
)
def test_mxfp8_forward(swiglu_limit, hidden, intermediate, macro):
    """Fused MXFP8 forward: E4M3 saves and both activation quantizations."""
    require_gpu()
    forward = MoKForwardMXFP8(swiglu_limit is not None)
    tokens, topk, mini, comm = 512, 2, 256, 4
    x, _, shared, routed, _, s = _setup(
        hidden, intermediate, 5197 + hidden + intermediate
    )
    routed_q = [quantize(w, transposed=False)[:2] for w in routed]
    combine = torch.zeros(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16)
    values = forward(
        x, [x.data_ptr()], combine, [combine.data_ptr()],
        shared[0], routed_q[0], shared[1], routed_q[1], shared[2], routed_q[2],
        s["peers"], s["schedule"], s["num_tokens"], s["counts"],
        topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    torch.cuda.synchronize()
    (
        (x_t, x_sc_t),
        _,
        (gate_q, gate_sc),
        _,
        (up_q, up_sc),
        _,
        (h_t, h_sc_t),
        y_shared,
        _,
    ) = values
    actual = s["actual"]
    rows = s["schedule"][:actual]
    valid = rows >= 0
    source = torch.zeros(actual, hidden, device="cuda", dtype=torch.bfloat16)
    source[valid] = x[rows[valid].long() // topk]
    xq, xs, xq_t, xs_t = quantize(source)
    x_deq = dequantize(xq, xs)
    w_g, w_u = dequantize(*routed_q[0]), dequantize(*routed_q[1])
    w_d = dequantize(*routed_q[2])
    gate_ref = torch.empty(actual, intermediate, device="cuda", dtype=torch.bfloat16)
    up_ref = torch.empty_like(gate_ref)
    y_ref = torch.empty(actual, hidden, device="cuda", dtype=torch.bfloat16)
    start = 0
    for e, n in enumerate(s["count_values"]):
        gate_ref[start : start + n] = (x_deq[start : start + n] @ w_g[e].T).bfloat16()
        up_ref[start : start + n] = (x_deq[start : start + n] @ w_u[e].T).bfloat16()
        start += n
    act = swiglu(gate_ref, up_ref, swiglu_limit)
    hq, hs, hq_t, hs_t = quantize(act)
    h_deq = dequantize(hq, hs)
    start = 0
    for e, n in enumerate(s["count_values"]):
        y_ref[start : start + n] = (h_deq[start : start + n] @ w_d[e].T).bfloat16()
        start += n
    expected_combine = torch.zeros_like(combine)
    expected_combine[rows[valid].long()] = y_ref[valid]
    sx = (x.float() @ shared[0].float().T).bfloat16()
    su = (x.float() @ shared[1].float().T).bfloat16()
    y_shared_ref = (
        swiglu(sx, su, swiglu_limit).float() @ shared[2].float().T
    ).bfloat16()
    assert _close(combine, expected_combine) == 0
    assert _close(y_shared, y_shared_ref) == 0
    # The ring keeps the first macrobatch as backward context. x quantization is
    # bitwise; GEMM-derived saves may differ only where BF16 rounding ties flip.
    r = min(actual, macro)
    assert torch.equal(x_t[:, :r], xq_t.view(torch.uint8)[:, :r])
    tiles = xs_t.view(hidden // 128, actual // 128, 32, 16)
    assert torch.equal(x_sc_t[:, : r // 128], tiles[:, : r // 128])
    gate_q_ref, gate_s_ref, _, _ = quantize(gate_ref, transposed=False)
    assert (
        gate_q[:r] == gate_q_ref.view(torch.uint8)[:r]
    ).float().mean().item() > 0.999
    assert (gate_sc[: r // 128] == gate_s_ref[: r // 128]).float().mean().item() > 0.999
    match = (h_t[:, :r] == hq_t.view(torch.uint8)[:, :r]).float().mean().item()
    assert match > 0.999


def _mxfp8_training(
    swiglu_limit,
    hidden,
    intermediate,
    macro,
    seed,
    repeat=1,
    experts=4,
    topk=2,
    fp32_wgrad=False,
    stats=None,
):
    """Mismatch counts of every returned gradient against the reference.

    The routed reference follows the MXFP8 recipe; the BF16 shared expert's
    weight gradients are checked against a BF16 reference of the shared MLP.
    ``fp32_wgrad`` adds every weight gradient into nonzero FP32 starting
    accumulators; the reference adds each macrobatch's FP32 product.

    The routed gate, up and dh are BF16 roundings of FP32 sums the kernel
    accumulates in another order: where a sum lies within the accumulation
    band of a BF16 rounding midpoint (``_tie_candidates``), the reference also
    evaluates the other neighbour and adds the exact effect of the E4M3 code
    flips it causes to the tolerance of the gradient elements they reach
    (``d_scores`` needs none: one neighbour of one of ``intermediate`` terms
    stays inside its gate). ``stats`` (a dict, optional) receives the
    candidate counts, the strict mismatch counts without that allowance and
    the allowance extents.
    """
    forward = MoKForwardMXFP8(swiglu_limit is not None)
    backward = MoKBackwardMXFP8(swiglu_limit is not None, fp32_wgrad)
    tokens, mini, comm = 512, 256, 4
    x, dy, shared, routed, scores, s = _setup(
        hidden,
        intermediate,
        seed + hidden + intermediate + macro,
        experts=experts,
        topk=topk,
    )
    q = [quantize(w) for w in routed]
    fwd_w = [(qq[0], qq[1]) for qq in q]
    bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
    options = dict(device="cuda", dtype=torch.bfloat16)
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer = torch.zeros(tokens * topk, hidden, **options)
    ds_peer = torch.zeros_like(scores)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    initial = accumulators = None
    if fp32_wgrad:
        initial = [
            torch.randn(w.shape, device="cuda") * 0.01 for w in (*shared, *routed)
        ]
        accumulators = [value.clone() for value in initial]

    def run():
        combine.zero_()
        dx_peer.zero_()
        ds_peer.zero_()
        if fp32_wgrad:
            for accumulator, start in zip(accumulators, initial, strict=True):
                accumulator.copy_(start)
        values = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        result = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *values[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        torch.cuda.synchronize()
        # Returned gradients only; macrobatch scratch has uninitialized padding.
        return tuple(t.clone() for t in (dx_peer, ds_peer, result[0], *result[9:]))

    first = run()
    for _ in range(repeat):
        again = run()
        assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    actual = s["actual"]
    rows = s["schedule"][:actual]
    valid = rows >= 0
    token = rows.clamp(min=0).long() // topk
    source = torch.where(valid[:, None], x[token], torch.zeros_like(x[token]))
    score = torch.where(
        valid,
        scores.flatten()[rows.clamp(min=0).long()],
        torch.zeros_like(rows, dtype=torch.float32),
    )
    xq, xs, xqt, xst = quantize(source)
    x_deq, xt_deq = dequantize(xq, xs), dequantize(xqt, xst)
    wg, wu = dequantize(q[0][0], q[0][1]), dequantize(q[1][0], q[1][1])
    wg_t, wu_t, wd_t = (dequantize(qq[2], qq[3]) for qq in q)
    segments, start = [], 0
    for e, n in enumerate(s["count_values"]):
        segments.append((e, start, start + n))
        start += n

    def routed_gemm(a, weights):
        """Per-expert ``a @ w.T``: the FP32 product the reference rounds, its FP64 value and the sum of |products|."""
        value = torch.empty(
            actual, weights.shape[1], device="cuda", dtype=torch.float32
        )
        exact = torch.empty(
            actual, weights.shape[1], device="cuda", dtype=torch.float64
        )
        magnitude = torch.empty_like(value)
        for e, lo, hi in segments:
            value[lo:hi] = a[lo:hi] @ weights[e].T
            exact[lo:hi] = a[lo:hi].double() @ weights[e].double().T
            magnitude[lo:hi] = a[lo:hi].abs() @ weights[e].abs().T
        return value, exact, magnitude

    gate_f32, gate_exact, gate_magnitude = routed_gemm(x_deq, wg)
    up_f32, up_exact, up_magnitude = routed_gemm(x_deq, wu)
    gate, up = gate_f32.bfloat16(), up_f32.bfloat16()
    dys = torch.where(
        valid[:, None],
        (dy[token].float() * score[:, None]).bfloat16(),
        torch.zeros_like(dy[token]),
    )
    dq, dsc, dqt, dsct = quantize(dys)
    d_deq, dt_deq = dequantize(dq, dsc), dequantize(dqt, dsct)
    dh_f32, dh_exact, dh_magnitude = routed_gemm(d_deq, wd_t)
    dh = dh_f32.bfloat16()

    def routed_backward(gate, up, dh):
        """The recipe's SwiGLU backward of BF16 gate/up/dh: dequantized E4M3
        d_gate/d_up in both orientations, the transposed activation
        quantization (the down weight-gradient operand) and the FP32 terms of
        the score gradient."""
        act = swiglu(gate, up, swiglu_limit)
        _, _, hqt, hst = quantize(act)
        g = dequantize(*quantize(gate, transposed=False)[:2])
        u = dequantize(*quantize(up, transposed=False)[:2])
        mask_g = mask_u = torch.ones_like(g)
        if swiglu_limit is not None:
            mask_g = (g <= swiglu_limit).float()
            mask_u = ((u >= -swiglu_limit) & (u <= swiglu_limit)).float()
            g, u = g.clamp(max=swiglu_limit), u.clamp(-swiglu_limit, swiglu_limit)
        sig = torch.sigmoid(g)
        silu = g * sig
        dg = (((1 - silu) * sig + silu) * u * dh.float() * mask_g).bfloat16()
        du = (silu * dh.float() * mask_u).bfloat16()
        dgq, dgs, dgqt, dgst = quantize(dg)
        duq, dus, duqt, dust = quantize(du)
        return dict(
            ht=dequantize(hqt, hst),
            dg=dequantize(dgq, dgs),
            du=dequantize(duq, dus),
            dgt=dequantize(dgqt, dgst),
            dut=dequantize(duqt, dust),
            ds_terms=dh.float() * silu * u,
        )

    chain = routed_backward(gate, up, dh)
    # Tie candidates: the backward re-evaluated with each GEMM-derived input
    # rounded to its other BF16 neighbour where that rounding is ambiguous.
    inputs = dict(gate=gate, up=up, dh=dh)
    sums = dict(
        gate=(gate_exact, gate_magnitude),
        up=(up_exact, up_magnitude),
        dh=(dh_exact, dh_magnitude),
    )
    alternates = []
    for name, (exact_sum, magnitude) in sums.items():
        ambiguous, other = _tie_candidates(inputs[name], exact_sum, magnitude)
        count = int(ambiguous.sum())
        if stats is not None:
            stats[f"ambiguous_{name}"] = count
        if count:
            alternates.append(routed_backward(**{**inputs, name: other}))
    allow_dw = [torch.zeros(w.shape, device="cuda") for w in routed]
    allow_dx = torch.zeros(actual, hidden, device="cuda")
    xt_abs, dt_abs = xt_deq.abs(), dt_deq.abs()
    for alt in alternates:
        delta = {k: (alt[k] - chain[k]).abs() for k in ("ht", "dg", "du", "dgt", "dut")}
        for e, lo, hi in segments:
            allow_dw[0][e] += delta["dgt"][:, lo:hi] @ xt_abs[:, lo:hi].T
            allow_dw[1][e] += delta["dut"][:, lo:hi] @ xt_abs[:, lo:hi].T
            allow_dw[2][e] += dt_abs[:, lo:hi] @ delta["ht"][:, lo:hi].T
            allow_dx[lo:hi] += (
                delta["dg"][lo:hi] @ wg_t[e].abs().T
                + delta["du"][lo:hi] @ wu_t[e].abs().T
            )
    inverse = torch.where(score > 0, 1 / score, torch.zeros_like(score))
    ds_route = chain["ds_terms"].sum(-1) * inverse
    dx_route = torch.empty(actual, hidden, **options)
    for e, lo, hi in segments:
        dx_route[lo:hi] = (
            chain["dg"][lo:hi] @ wg_t[e].T + chain["du"][lo:hi] @ wu_t[e].T
        ).bfloat16()
    if fp32_wgrad:
        dw = [value.clone() for value in initial[3:]]
    else:
        dw = [torch.zeros_like(w) for w in routed]
    for e, lo, hi in segments:
        first_part = True
        for m_lo in range(0, actual, macro):
            k_lo, k_hi = max(lo, m_lo), min(hi, m_lo + macro)
            if k_lo >= k_hi:
                continue
            exact = [
                chain["dgt"][:, k_lo:k_hi] @ xt_deq[:, k_lo:k_hi].T,
                chain["dut"][:, k_lo:k_hi] @ xt_deq[:, k_lo:k_hi].T,
                dt_deq[:, k_lo:k_hi] @ chain["ht"][:, k_lo:k_hi].T,
            ]
            for dest, part in zip(dw, exact, strict=True):
                if fp32_wgrad:
                    dest[e] += part
                elif first_part:
                    dest[e] = part.bfloat16()
                else:
                    # Native BF16 accumulation across macrobatches.
                    dest[e] = (dest[e].float() + part.bfloat16().float()).bfloat16()
            first_part = False
    # BF16 shared expert over every local token (unscaled dY), as the shared
    # BF16 GEMMs and SwiGLU tasks compute it.
    s_gate = (x.float() @ shared[0].float().T).bfloat16()
    s_up = (x.float() @ shared[1].float().T).bfloat16()
    s_act = swiglu(s_gate, s_up, swiglu_limit)
    s_g, s_u = s_gate.float(), s_up.float()
    s_mask_g = s_mask_u = torch.ones_like(s_g)
    if swiglu_limit is not None:
        s_mask_g = (s_g <= swiglu_limit).float()
        s_mask_u = ((s_u >= -swiglu_limit) & (s_u <= swiglu_limit)).float()
        s_g, s_u = s_g.clamp(max=swiglu_limit), s_u.clamp(-swiglu_limit, swiglu_limit)
    s_sig = torch.sigmoid(s_g)
    s_silu = s_g * s_sig
    s_dh = (dy.float() @ shared[2].float()).bfloat16()
    s_dg = (((1 - s_silu) * s_sig + s_silu) * s_u * s_dh.float() * s_mask_g).bfloat16()
    s_du = (s_silu * s_dh.float() * s_mask_u).bfloat16()
    dw_shared = [
        s_dg.float().T @ x.float(),
        s_du.float().T @ x.float(),
        dy.float().T @ s_act.float(),
    ]
    if fp32_wgrad:
        dw_shared = [a + b for a, b in zip(initial[:3], dw_shared, strict=True)]
    else:
        dw_shared = [part.bfloat16() for part in dw_shared]
    expected_dx = torch.zeros_like(dx_peer)
    expected_dx[rows[valid].long()] = dx_route[valid]
    allow_dx_peer = torch.zeros(dx_peer.shape, device="cuda")
    allow_dx_peer[rows[valid].long()] = allow_dx[valid]
    expected_ds = torch.zeros_like(ds_peer).flatten()
    expected_ds[rows[valid].long()] = ds_route[valid]
    dx_peer, ds_peer, _, dws_gate, dw_gate, dws_up, dw_up, dws_down, dw_down = first
    checks = dict(
        dx_routed=(dx_peer, expected_dx, allow_dx_peer),
        d_scores=(ds_peer.flatten(), expected_ds, None),
        dw_gate=(dw_gate, dw[0], allow_dw[0]),
        dw_up=(dw_up, dw[1], allow_dw[1]),
        dw_down=(dw_down, dw[2], allow_dw[2]),
        dw_shared_gate=(dws_gate, dw_shared[0], None),
        dw_shared_up=(dws_up, dw_shared[1], None),
        dw_shared_down=(dws_down, dw_shared[2], None),
    )
    if stats is not None:
        for name, (a, b, allowance) in checks.items():
            stats[f"strict_{name}"] = _close(a, b)
            if allowance is not None:
                bound = 1e-2 + 1e-2 * b.float().abs()
                stats[f"allowed_{name}"] = int((allowance > 0).sum())
                stats[f"max_allowance_{name}"] = float(allowance.max())
                stats[f"max_allowance_over_bound_{name}"] = float(
                    (allowance / bound).max()
                )
    return {name: _close(a, b, allowance) for name, (a, b, allowance) in checks.items()}


@LIMITS
@pytest.mark.parametrize(
    "hidden,intermediate,macro", [(512, 256, 768), (512, 512, 1536), (512, 256, 512)]
)
def test_mxfp8_training(swiglu_limit, hidden, intermediate, macro):
    """Fused MXFP8 forward+backward against the exact quantization-aware reference.

    BF16 rounding ties of the GEMM-derived gate, up and dh are covered by the
    reference's tie candidates (``_mxfp8_training``). Reruns are bitwise.
    """
    require_gpu()
    mismatches = _mxfp8_training(swiglu_limit, hidden, intermediate, macro, seed=7000)
    assert all(v == 0 for v in mismatches.values()), mismatches


@LIMITS
@pytest.mark.parametrize(
    "hidden,intermediate,macro", [(512, 256, 768), (512, 512, 1536), (512, 256, 512)]
)
def test_mxfp8_training_fp32_wgrad(swiglu_limit, hidden, intermediate, macro):
    """MXFP8 training with FP32 weight-gradient accumulation onto nonzero starts.

    Shared (BF16) and routed (MXFP8) weight gradients are both added into the
    caller's FP32 accumulators; the shapes include multiple macrobatches.
    """
    require_gpu()
    mismatches = _mxfp8_training(
        swiglu_limit, hidden, intermediate, macro, seed=7000, fp32_wgrad=True
    )
    assert all(v == 0 for v in mismatches.values()), mismatches


@pytest.mark.parametrize("experts", [36, 9], ids=["ep8-local36", "ep32-local9"])
def test_mxfp8_training_glm_flash_local_experts(experts):
    """GLM-5.3-Flash local expert counts (288 experts at EP8/EP32), top-8, clamped.

    More routed rows raise the chance of an E4M3 rounding-boundary flip
    between the kernel and the PyTorch reference, so a few isolated
    elements may differ; the mismatch fraction must stay negligible.
    """
    require_gpu()
    mismatches = _mxfp8_training(
        0.75, 512, 256, 2048, seed=7000, experts=experts, topk=8
    )
    sizes = dict(
        dx_routed=512 * 8 * 512,
        d_scores=512 * 8,
        dw_gate=experts * 256 * 512,
        dw_up=experts * 256 * 512,
        dw_down=experts * 512 * 256,
        dw_shared_gate=256 * 512,
        dw_shared_up=256 * 512,
        dw_shared_down=512 * 256,
    )
    assert all(mismatches[k] <= 1e-4 * sizes[k] for k in sizes), mismatches


class _LaunchRecorder:
    """Stands in for a prepared kernel: records the launch keywords, launches nothing."""

    def __init__(self):
        self.calls = []

    def launch(self, *, grid, **kwargs):
        self.calls.append((grid, kwargs))

    def last(self, name):
        return next(kwargs for _, kwargs in reversed(self.calls) if name in kwargs)


def _check_counters(kwargs, expected):
    """Counter keywords ``expected`` (name -> numel) of one recorded fused launch:
    CUDA, contiguous, 1-D, int32 (uint32 for ``combine_next``), the exact
    numel, zero, 128-byte aligned, pairwise disjoint on one storage."""
    views = {name: kwargs[name] for name in expected}
    assert len({v.untyped_storage().data_ptr() for v in views.values()}) == 1
    for name, numel in expected.items():
        view = views[name]
        assert view.is_cuda and view.is_contiguous() and view.dim() == 1, name
        assert view.dtype == (torch.uint32 if name == "combine_next" else torch.int32)
        assert view.numel() == numel, (name, view.numel(), numel)
        assert torch.count_nonzero(view.view(torch.int32)).item() == 0, name
        assert view.data_ptr() % 128 == 0, name
    ranges = sorted(
        (v.data_ptr(), v.data_ptr() + 4 * v.numel()) for v in views.values()
    )
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False))


def test_mxfp8_launch_counters_are_one_slab_per_launch():
    """At the launch boundary the MXFP8 launchers hand the kernels the counter
    names, dtypes and extents they always did, as views of one zero slab per
    launch (``combine_next`` the uint32 view of its element); a warm call
    dispatches one ``zeros`` and one split for them and no fill, zeroing or
    copy."""
    require_gpu()
    from collections import Counter

    from torch.utils._python_dispatch import TorchDispatchMode

    hidden, intermediate, macro, tokens, topk, mini, comm = (
        512,
        256,
        768,
        512,
        2,
        256,
        4,
    )
    x, dy, shared, routed, scores, s = _setup(hidden, intermediate, 2027)
    q = [quantize(w) for w in routed]
    fwd_w = [(qq[0], qq[1]) for qq in q]
    bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    capacity = s["peers"].numel()
    options = dict(device="cuda", dtype=torch.bfloat16)
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer, ds_peer = torch.zeros_like(combine), torch.zeros_like(scores)
    forward, recompute, backward = (
        MoKForwardMXFP8(),
        MoKRecomputeMXFP8(),
        MoKBackwardMXFP8(),
    )
    for launcher in (forward, recompute, backward):
        launcher.module = launcher.layout_module = _LaunchRecorder()
    backward.zero_module = backward.module

    def run():
        saved = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, None, comm, macro, mini,
        )  # fmt: skip
        recompute(
            x, [x.data_ptr()], shared[0], fwd_w[0], shared[1], fwd_w[1],
            *schedule, topk, None, comm, macro, mini,
        )  # fmt: skip
        backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *saved[:7], x, [x.data_ptr()], *schedule,
            topk, None, comm, macro, mini,
        )  # fmt: skip

    run()  # cold: peer tables and geometry
    ops = []

    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            ops.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    with Mode():
        forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, None, comm, macro, mini,
        )  # fmt: skip
    counted = Counter(ops)
    assert counted["zeros"] == 1 and counted["split_with_sizes"] == 1, counted
    assert not (set(counted) & {"fill_", "zero_", "copy_"}), counted
    run()
    minis, macros = (capacity + mini - 1) // mini, (capacity + macro - 1) // macro
    shared_rows, routed_rows = tokens // 256, capacity // 256
    ib = intermediate // 256
    _check_counters(
        forward.module.last("x_ready"),
        dict(
            x_ready=minis,
            gate_ready=(shared_rows + routed_rows) * ib,
            hidden_ready=shared_rows + routed_rows,
            y_ready=minis,
            y_done=capacity // 128,
            combine_next=1,
        ),
    )
    routed_capacity = min(capacity, macro)
    _check_counters(
        recompute.module.last("x_ready"),
        dict(
            x_ready=(routed_capacity + mini - 1) // mini,
            gate_ready=(shared_rows + routed_capacity // 256) * ib,
            hidden_ready=shared_rows + routed_capacity // 256,
        ),
    )
    _check_counters(
        backward.module.last("dy_ready"),
        dict(
            dy_ready=minis,
            dx_ready=minis,
            replay_x=minis,
            dh_ready=(shared_rows + routed_rows) * ib,
            dg_ready=shared_rows + routed_rows,
            replay_gu=routed_rows * ib,
            replay_h=routed_rows,
            buffers_done=macros,
            weight_ready=macros,
        ),
    )
    launch = forward.module.last("x_ready")
    for name, weight in (("wg_q", fwd_w[0][0]), ("wg_sc", fwd_w[0][1])):
        assert launch[name].dtype == torch.uint8
        assert launch[name].data_ptr() == weight.data_ptr()
    assert launch["wg_shared"].shape == (1, *shared[0].shape)
    assert launch["wg_shared"].data_ptr() == shared[0].data_ptr()
    assert (
        len(forward._geometry)
        == len(recompute._geometry)
        == len(backward._geometry)
        == 1
    )


@LIMITS
def test_mxfp8_launchers_accept_a_precomputed_layout(swiglu_limit):
    """The MXFP8 launchers derive the expert layout themselves (``layout=None``)
    or read the one passed to them; both paths are bitwise equal."""
    require_gpu()
    clamped = swiglu_limit is not None
    forward, recompute = MoKForwardMXFP8(clamped), MoKRecomputeMXFP8(clamped)
    backward = MoKBackwardMXFP8(clamped)
    tokens, hidden, intermediate, topk, mini, comm, macro = (
        512,
        512,
        256,
        2,
        256,
        4,
        768,
    )
    x, dy, shared, routed, scores, s = _setup(hidden, intermediate, 811)
    q = [quantize(w) for w in routed]
    fwd_w = [(qq[0], qq[1]) for qq in q]
    bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    capacity = s["peers"].numel()
    layout = kernels.expert_layout(
        forward.layout_module, s["counts"], capacity, s["peers"]
    )
    options = dict(device="cuda", dtype=torch.bfloat16)
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer, ds_peer = torch.zeros_like(combine), torch.zeros_like(scores)
    rows = min(s["actual"], macro)

    def run(layout_arg):
        combine.zero_()
        dx_peer.zero_()
        ds_peer.zero_()
        saved = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, swiglu_limit, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        rebuilt = recompute(
            x, [x.data_ptr()], shared[0], fwd_w[0], shared[1], fwd_w[1],
            *schedule, topk, swiglu_limit, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        result = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *saved[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        torch.cuda.synchronize()
        pieces = [combine, dx_peer, ds_peer, result[0], *result[9:], saved[7]]
        for context in (saved, rebuilt):
            for index, transposed in ((0, True), (2, False), (4, False), (6, True)):
                data, scales = context[index]
                if transposed:  # [features, tokens] data, [feature tiles, token tiles]
                    pieces += [data[:, :rows], scales[:, : rows // 128]]
                else:
                    pieces += [data[:rows], scales[: rows // 128]]
            pieces += [context[index] for index in (1, 3, 5)]
        return [piece.clone() for piece in pieces]

    first = run(None)
    for layout_arg in (None, layout, layout):
        again = run(layout_arg)
        assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))


@LIMITS
@pytest.mark.parametrize("macro", [768, 4096], ids=["replay", "resident"])
def test_mxfp8_recompute_context(swiglu_limit, macro):
    """MXFP8 context-only recompute equals the saved context; backward is bitwise."""
    require_gpu()
    clamped = swiglu_limit is not None
    forward, recompute = MoKForwardMXFP8(clamped), MoKRecomputeMXFP8(clamped)
    backward = MoKBackwardMXFP8(clamped)
    tokens, hidden, intermediate, topk, mini, comm = 512, 512, 256, 2, 256, 4
    x, dy, shared, routed, scores, s = _setup(hidden, intermediate, 811)
    q = [quantize(w) for w in routed]
    fwd_w = [(qq[0], qq[1]) for qq in q]
    bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    combine = torch.zeros(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16)
    saved = forward(
        x, [x.data_ptr()], combine, [combine.data_ptr()],
        shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    rebuilt = recompute(
        x, [x.data_ptr()], shared[0], fwd_w[0], shared[1], fwd_w[1],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    rows = min(s["actual"], macro)
    for index, transposed in ((0, True), (2, False), (4, False), (6, True)):
        (a, a_sc), (b, b_sc) = saved[index], rebuilt[index]
        if transposed:  # [features, tokens] data, [feature tiles, token tiles] scales
            assert torch.equal(a[:, :rows], b[:, :rows]), index
            assert torch.equal(a_sc[:, : rows // 128], b_sc[:, : rows // 128]), index
        else:
            assert torch.equal(a[:rows], b[:rows]), index
            assert torch.equal(a_sc[: rows // 128], b_sc[: rows // 128]), index
    for index in (1, 3, 5):
        assert torch.equal(saved[index], rebuilt[index]), index

    def gradients(context):
        dx_peer = torch.zeros(
            tokens * topk, hidden, device="cuda", dtype=torch.bfloat16
        )
        ds_peer = torch.zeros_like(scores)
        values = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *context[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        return (dx_peer, ds_peer, values[0], *values[9:])

    for a, b in zip(gradients(saved), gradients(rebuilt), strict=True):
        assert torch.equal(a, b)
