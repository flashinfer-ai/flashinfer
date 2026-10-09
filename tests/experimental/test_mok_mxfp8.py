# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU native MXFP8 MoK checks against a quantization-aware reference.

The reference follows the MoK recipe in schedule order: x, the SwiGLU
activation, the score-scaled dY and d_gate/d_up are MXFP8 (32-element blocks
along K), routed weights are caller-prequantized, GEMMs accumulate in FP32 on
dequantized operands and round to BF16, and the shared expert stays BF16.
"""

import pytest
import torch

from flashinfer.experimental.cake_mok_bf16._kernels import (
    MoKBackwardMXFP8,
    MoKForwardMXFP8,
    MoKRecomputeMXFP8,
)
from flashinfer.mok import quantize_mok_mxfp8_weights
from tests.experimental._mok_reference import (
    mxfp8_dequantize as dequantize,
    mxfp8_quantize as quantize,
    require_gpu,
    single_rank_schedule,
    swiglu,
)

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


def _close(a, b):
    d = (a.float() - b.float()).abs()
    return int((d > 1e-2 + 1e-2 * b.float().abs()).sum())


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
    swiglu_limit, hidden, intermediate, macro, seed, repeat=1, experts=4, topk=2
):
    forward = MoKForwardMXFP8(swiglu_limit is not None)
    backward = MoKBackwardMXFP8(swiglu_limit is not None)
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

    def run():
        combine.zero_()
        dx_peer.zero_()
        ds_peer.zero_()
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
        )  # fmt: skip
        torch.cuda.synchronize()
        # Returned gradients only; macrobatch scratch has uninitialized padding.
        return (dx_peer.clone(), ds_peer.clone(), result[0], *result[9:])

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
    gate = torch.empty(actual, intermediate, **options)
    up = torch.empty_like(gate)
    for e, lo, hi in segments:
        gate[lo:hi] = (x_deq[lo:hi] @ wg[e].T).bfloat16()
        up[lo:hi] = (x_deq[lo:hi] @ wu[e].T).bfloat16()
    act = swiglu(gate, up, swiglu_limit)
    _, _, hqt, hst = quantize(act)
    ht_deq = dequantize(hqt, hst)
    dys = torch.where(
        valid[:, None],
        (dy[token].float() * score[:, None]).bfloat16(),
        torch.zeros_like(dy[token]),
    )
    dq, dsc, dqt, dsct = quantize(dys)
    d_deq, dt_deq = dequantize(dq, dsc), dequantize(dqt, dsct)
    dh = torch.empty(actual, intermediate, **options)
    for e, lo, hi in segments:
        dh[lo:hi] = (d_deq[lo:hi] @ wd_t[e].T).bfloat16()
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
    inverse = torch.where(score > 0, 1 / score, torch.zeros_like(score))
    ds_route = (dh.float() * silu * u).sum(-1) * inverse
    dgq, dgs, dgqt, dgst = quantize(dg)
    duq, dus, duqt, dust = quantize(du)
    dg_deq, du_deq = dequantize(dgq, dgs), dequantize(duq, dus)
    dgt_deq, dut_deq = dequantize(dgqt, dgst), dequantize(duqt, dust)
    dx_route = torch.empty(actual, hidden, **options)
    for e, lo, hi in segments:
        dx_route[lo:hi] = (
            dg_deq[lo:hi] @ wg_t[e].T + du_deq[lo:hi] @ wu_t[e].T
        ).bfloat16()
    dw = [torch.zeros_like(w) for w in routed]
    for e, lo, hi in segments:
        first_part = True
        for m_lo in range(0, actual, macro):
            k_lo, k_hi = max(lo, m_lo), min(hi, m_lo + macro)
            if k_lo >= k_hi:
                continue
            parts = [
                (dgt_deq[:, k_lo:k_hi] @ xt_deq[:, k_lo:k_hi].T).bfloat16(),
                (dut_deq[:, k_lo:k_hi] @ xt_deq[:, k_lo:k_hi].T).bfloat16(),
                (dt_deq[:, k_lo:k_hi] @ ht_deq[:, k_lo:k_hi].T).bfloat16(),
            ]
            for dest, part in zip(dw, parts, strict=True):
                dest[e] = (
                    part if first_part else (dest[e].float() + part.float()).bfloat16()
                )
            first_part = False
    expected_dx = torch.zeros_like(dx_peer)
    expected_dx[rows[valid].long()] = dx_route[valid]
    expected_ds = torch.zeros_like(ds_peer).flatten()
    expected_ds[rows[valid].long()] = ds_route[valid]
    dx_peer, ds_peer, _, _, dw_gate, _, dw_up, _, dw_down = first
    return dict(
        dx_routed=_close(dx_peer, expected_dx),
        d_scores=_close(ds_peer.flatten(), expected_ds),
        dw_gate=_close(dw_gate, dw[0]),
        dw_up=_close(dw_up, dw[1]),
        dw_down=_close(dw_down, dw[2]),
    )


@LIMITS
@pytest.mark.parametrize(
    "hidden,intermediate,macro", [(512, 256, 768), (512, 512, 1536), (512, 256, 512)]
)
def test_mxfp8_training(swiglu_limit, hidden, intermediate, macro):
    """Fused MXFP8 forward+backward against the exact quantization-aware reference.

    The reference evaluates SwiGLU backward with PyTorch transcendentals; for
    some inputs an E4M3 block of d_gate/d_up lands on the other side of a
    rounding boundary. This seed has no such flip. Reruns are bitwise.
    """
    require_gpu()
    mismatches = _mxfp8_training(swiglu_limit, hidden, intermediate, macro, seed=7000)
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
    )
    assert all(mismatches[k] <= 1e-4 * sizes[k] for k in sizes), mismatches


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
