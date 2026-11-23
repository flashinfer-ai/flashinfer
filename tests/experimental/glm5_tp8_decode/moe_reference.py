# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Independent W8A8 MoE references for q_len one and four.

The expert math, Q128 quantization and rounding boundaries match both kernels.
Approximate-sigmoid routing is checked against its analytical error bound; this
is not a generic allclose tolerance. The references never call the kernels.
"""

from fractions import Fraction
import math
import struct

import torch

FP8 = torch.float8_e4m3fn
HIDDEN, INTER_PER_RANK, EXPERTS, TOPK = 6144, 256, 256, 8
ROUTED_SCALE = 2.5
EPS = 1e-5
SIGMOID_ABS_ERR = 2.0**-12


def bitwise(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False

    def bytes_view(value):
        value = value.contiguous()
        if value.ndim == 0:
            value = value.reshape(1)
        # A length-one slice can be "contiguous" while retaining stride6144.
        # Reinterpreting a different element width still requires last stride1.
        if value.stride(-1) != 1:
            value = value.clone(memory_format=torch.contiguous_format)
        return value.view(torch.uint8)

    return torch.equal(bytes_view(a), bytes_view(b))


def rmsnorm(x, gamma):
    xf = x.float().cpu()
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    # Both q1 and q4 multiply gamma by rstd before multiplying x. Moving
    # parentheses can cross a BF16 rounding midpoint even with the same rstd.
    return (xf * (gamma.float().cpu()[None, :] * rms)).to(torch.bfloat16)


def _f32_from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _rsqrt_candidates(argument):
    """Float32 results within CUDA 13.2's documented rsqrtf 2-ULP accuracy.

    The CUDA guide measures error from the correctly rounded float32 result,
    and describes this as an observed/tested bound, not an exhaustive guarantee:
    https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html
    """
    if argument == math.inf:
        return [0.0]  # CUDA rsqrtf(+infinity), e.g. FP32 square-sum overflow.
    assert math.isfinite(argument) and argument > 0
    bits = struct.unpack("<I", struct.pack("<f", 1 / math.sqrt(argument)))[0]
    exact_argument = Fraction(argument)
    # Correct a possible double-rounding tie in the FP64 starting estimate.
    # Comparing 1 with argument*midpoint**2 is exact rational arithmetic.
    while True:
        center = _f32_from_bits(bits)
        lower = (_f32_from_bits(bits - 1) + center) / 2
        upper = (center + _f32_from_bits(bits + 1)) / 2
        low_square = exact_argument * Fraction(lower) ** 2
        high_square = exact_argument * Fraction(upper) ** 2
        if low_square > 1 or (low_square == 1 and bits & 1):
            bits -= 1
        elif high_square < 1 or (high_square == 1 and bits & 1):
            bits += 1
        else:
            break
    ulp = math.ldexp(1.0, math.frexp(center)[1] - 24)
    # At a power of two the lower spacing is half the upper spacing: RN +/-2
    # bit patterns would omit legal values. Enumerate then filter by distance.
    return [
        value
        for offset in range(-8, 9)
        if abs((value := _f32_from_bits(bits + offset)) - center) <= 2 * ulp
    ]


def routed_norm_candidates(x, gamma):
    """Independent q4 routed RMSNorm candidates; no kernel outputs are inputs.

    Reconstruct the specified FP32 reduction order, then allow only the scalar
    rsqrtf uncertainty above. The shared-expert normalization has a different
    lane assignment, so its observable scratch is not a router oracle.
    """
    xf = x.detach().cpu().double()
    gf = gamma.detach().cpu().float()
    assert x.dtype == torch.bfloat16 and xf.ndim == 2 and xf.shape[1] == HIDDEN
    assert bool(torch.isfinite(xf).all()) and bool(torch.isfinite(gf).all())
    pairs = xf.reshape(xf.shape[0], -1, 2)
    # BF16 products have at most 16 significant bits. FP64 followed by FP32
    # rounding reproduces the positive pair FMA, including its rounded hi^2.
    high_square = pairs[..., 1].square().float().double()
    pair_sums = (pairs[..., 0].square() + high_square).float()
    # Each lane owns three 8-element chunks, at chunk = lane + 256*j.
    lane_pairs = (
        pair_sums.reshape(-1, 3, 256, 4).permute(0, 2, 1, 3).reshape(-1, 256, 12)
    )
    accum = torch.zeros(xf.shape[0], 256, dtype=torch.float32)
    for index in range(12):
        accum = accum + lane_pairs[..., index]
    warp = accum.reshape(-1, 8, 32)
    for offset in (16, 8, 4, 2, 1):
        warp = warp[..., :offset] + warp[..., offset : 2 * offset]
    p = warp[..., 0]
    total = ((p[:, 0] + p[:, 4]) + (p[:, 2] + p[:, 6])) + (
        (p[:, 1] + p[:, 5]) + (p[:, 3] + p[:, 7])
    )
    arguments = (total.double() / HIDDEN).float() + torch.tensor(
        EPS, dtype=torch.float32
    )
    candidates = []
    for token, argument in enumerate(arguments.tolist()):
        values = []
        for scalar in _rsqrt_candidates(argument):
            value = (xf[token].float() * (gf * scalar)).bfloat16()
            if not any(bitwise(value, other) for other in values):
                values.append(value)
        candidates.append(torch.stack(values))
    return candidates


def check_router_candidates(logits, candidates, router):
    """Require one whole-row norm candidate to explain every expert logit."""
    actual = logits.detach().cpu().double()
    weights = router.detach().cpu().double()
    assert len(candidates) == actual.shape[0] and actual.shape[1] == weights.shape[0]
    assert bool(torch.isfinite(actual).all()), "nonfinite router logits"
    gamma64 = 64 * 2.0**-24 / (1 - 64 * 2.0**-24)
    report = []
    for token, norms in enumerate(candidates):
        norms = norms.double()
        expected = norms @ weights.T
        bound = gamma64 * (norms.abs() @ weights.abs().T) + 1e-30
        ratio = (actual[token] - expected).abs() / bound
        # max-before-min is essential: per-expert candidate choice would accept
        # a row that no possible scalar rstd can produce.
        worst = ratio.amax(dim=1)
        best = int(worst.argmin())
        row = {
            "token": token,
            "norm_candidates": norms.shape[0],
            "router_max_error_over_bound": float(worst[best]),
        }
        assert bool(worst[best] <= 1), row
        report.append(row)
    return report


def check_router_logits(logits, x, gamma, router):
    return check_router_candidates(logits, routed_norm_candidates(x, gamma), router)


def reduce_ranks(partials, residual):
    acc = torch.zeros(partials[0].shape, dtype=torch.float32)
    for p in partials:
        acc += p.float().cpu()
    acc += residual.float().cpu()
    return acc.to(torch.bfloat16)


def q128(x):
    xb = x.float().reshape(*x.shape[:-1], x.shape[-1] // 128, 128)
    scale = xb.abs().amax(-1).clamp_min(0.0001) * torch.tensor(
        1 / 448, dtype=torch.float32, device=x.device
    )
    inv = torch.ones_like(scale) / scale
    return ((xb * inv.unsqueeze(-1)).to(FP8).reshape(x.shape), scale)


def floored_ulp(actual, ref, budget=2):
    scale = ref.float().abs().clamp_min(1 / 16)
    ulp = torch.exp2(torch.floor(torch.log2(scale)) - 7)
    err = (actual.float() - ref.float()).abs()
    units = err / ulp
    assert bool(torch.isfinite(units).all()), "nonfinite MoE comparison"
    report = {
        "max_abs": err.max().item(),
        "max_floor_ULP": units.max().item(),
        "over_budget": int((units > budget).sum()),
        "elements": actual.numel(),
        "bitwise_fraction": float((actual == ref).float().mean()),
    }
    assert report["over_budget"] == 0, report
    return report


def bank_projection(weights, banks, projection):
    """Stack one projection of the given banks from rank-local checkpoint tensors.

    ``weights`` holds the rank-local MoE tensors under their checkpoint names.
    Bank 0 is the shared expert and bank ``e + 1`` routed expert ``e``. Returns
    the FP8 codes ``[n, ...]`` and their FP32 block scales ``[n, ...]``.
    """
    names = [
        "mlp.shared_experts" if bank == 0 else f"mlp.experts.{bank - 1}"
        for bank in banks.tolist()
    ]
    codes = torch.stack(
        [weights[f"{name}.{projection}.weight"].view(torch.uint8) for name in names]
    )
    scales = torch.stack(
        [weights[f"{name}.{projection}.weight_scale_inv"] for name in names]
    )
    return codes.view(FP8), scales


def _w8a8_gate_up(q, s, weights, banks):
    n = banks.numel()
    outputs = []
    for projection in ("gate_proj", "up_proj"):
        codes, scales = bank_projection(weights, banks, projection)
        w = codes.float().reshape(n, 256, 48, 128)
        ws = scales.repeat_interleave(128, dim=1)
        total = torch.zeros(n, 256, device=q.device)
        for kb in range(48):
            part = torch.bmm(
                w[:, :, kb, :].contiguous(), q[:, kb, :].unsqueeze(-1)
            ).squeeze(-1)
            total += part * (ws[:, :, kb] * s[:, kb, None])
        outputs.append(total)
    gate, up = outputs
    return (torch.nn.functional.silu(gate) * up).bfloat16()


def reference_mid(norm, ids, weights):
    q_len = norm.shape[0]
    q, s = q128(norm)
    q = q.float().reshape(q_len, 48, 128)
    banks = torch.cat(
        [torch.zeros(q_len, 1, dtype=torch.int64, device=norm.device), ids.long() + 1],
        dim=1,
    ).reshape(-1)
    acts = q[:, None].expand(q_len, 9, 48, 128).reshape(q_len * 9, 48, 128)
    scales = s[:, None].expand(q_len, 9, 48).reshape(q_len * 9, 48)
    return _w8a8_gate_up(acts, scales, weights, banks).reshape(q_len, 9, 256)


def shared_mid(norm, weights):
    q_len = norm.shape[0]
    q, s = q128(norm)
    banks = torch.zeros(q_len, dtype=torch.int64, device=norm.device)
    return _w8a8_gate_up(q.float().reshape(q_len, 48, 128), s, weights, banks)


def reference_partial(mid, ids, probs, weights):
    q_len = mid.shape[0]
    q, dscale = q128(mid)
    q = q.float().reshape(q_len * 9, 2, 128)
    banks = torch.cat(
        [torch.zeros(q_len, 1, dtype=torch.int64, device=mid.device), ids.long() + 1],
        dim=1,
    ).reshape(q_len * 9)
    probability = torch.cat(
        [torch.ones(q_len, 1, device=mid.device), probs], dim=1
    ).reshape(q_len * 9)
    codes, scales = bank_projection(weights, banks, "down_proj")
    w = codes.float().reshape(q_len * 9, 6144, 2, 128)
    ws = scales.repeat_interleave(128, dim=1)
    acc = torch.zeros(q_len * 9, 6144, device=mid.device)
    for kb in range(2):
        part = torch.bmm(w[:, :, kb, :].contiguous(), q[:, kb].unsqueeze(-1)).squeeze(
            -1
        )
        acc += part * (
            ws[:, :, kb]
            * probability[:, None]
            * dscale.reshape(q_len * 9, 2)[:, kb, None]
        )
    out = torch.zeros(q_len, 6144, device=mid.device)
    for slot in range(9):
        out += acc.reshape(q_len, 9, 6144)[:, slot]
    return out.bfloat16()


def reference_partial_terms(mid, ids, weights):
    q_len = mid.shape[0]
    q, dscale = q128(mid)
    q = q.float().reshape(q_len * 9, 2, 128)
    banks = torch.cat(
        [torch.zeros(q_len, 1, dtype=torch.int64, device=mid.device), ids.long() + 1],
        dim=1,
    ).reshape(q_len * 9)
    codes, scales = bank_projection(weights, banks, "down_proj")
    w = codes.float().reshape(q_len * 9, 6144, 2, 128)
    ws = scales.repeat_interleave(128, dim=1)
    acc = torch.zeros(q_len * 9, 6144, device=mid.device)
    for kb in range(2):
        part = torch.bmm(w[:, :, kb, :].contiguous(), q[:, kb].unsqueeze(-1)).squeeze(
            -1
        )
        acc += part * (ws[:, :, kb] * dscale.reshape(q_len * 9, 2)[:, kb, None])
    return acc.reshape(q_len, 9, 6144)


def kernel_logits(scores, q_len):
    width = 2 * q_len
    return scores[:, :width].reshape(q_len, 128 // q_len, width).reshape(q_len, 256)


def candidate_selections(choice, tol, limit=64):
    import itertools

    order = torch.sort(choice, descending=True, stable=True).indices
    thr = float(choice[order[TOPK - 1]])
    sure = [int(e) for e in order if float(choice[e]) > thr + 2 * tol]
    band = [int(e) for e in order if abs(float(choice[e]) - thr) <= 2 * tol]
    first = [int(e) for e in order[:TOPK]]
    out = [first]
    for combo in itertools.combinations(band, TOPK - len(sure)):
        sel = sure + list(combo)
        if set(sel) != set(first):
            out.append(sel)
        if len(out) >= limit:
            break
    return out


def routing_weights(sigma, ids):
    s = torch.gather(sigma, 1, ids.long())
    total = s.sum(-1, keepdim=True)
    probs = ROUTED_SCALE * s / total
    bound = ROUTED_SCALE * SIGMOID_ABS_ERR * (1 + TOPK * s / total) / total
    return (probs, bound)


def route_reference(norm, router, bias):
    logits = norm.double() @ router.double().T
    sigma = torch.sigmoid(logits)
    return (sigma, sigma + bias.double()[None, :])


def check_routing(ids, probs, sigma, choice):
    ids_c, probs_c = (ids.long().cpu(), probs.double().cpu())
    sigma, choice = (sigma.cpu(), choice.cpu())
    tol = 2 * SIGMOID_ABS_ERR
    report = {"swapped_at_boundary": 0}
    for t in range(ids_c.shape[0]):
        sel = ids_c[t]
        assert (
            len(set(sel.tolist())) == TOPK
            and int(sel.min()) >= 0
            and (int(sel.max()) < EXPERTS)
        ), ("ids", t, sel.tolist())
        order = torch.sort(choice[t], descending=True, stable=True).indices
        ref = set(order[:TOPK].tolist())
        thr = float(choice[t, order[TOPK - 1]])
        extra, missing = (set(sel.tolist()) - ref, ref - set(sel.tolist()))
        assert all((float(choice[t, e]) >= thr - tol for e in extra)), (
            "selection",
            t,
            sorted(extra),
            thr,
        )
        assert all((float(choice[t, m]) <= thr + tol for m in missing)), (
            "selection",
            t,
            sorted(missing),
            thr,
        )
        report["swapped_at_boundary"] += len(extra)
        s = sigma[t, sel]
        expect = ROUTED_SCALE * s / s.sum()
        bound = (
            ROUTED_SCALE * SIGMOID_ABS_ERR * (1 + TOPK * s / s.sum()) / s.sum() + 1e-06
        )
        err = (probs_c[t] - expect).abs()
        assert bool((err <= bound).all()), (
            "probs",
            t,
            float(err.max()),
            float(bound.min()),
        )
        assert abs(float(probs_c[t].sum()) - ROUTED_SCALE) <= 1e-05, (
            "probs sum",
            t,
            float(probs_c[t].sum()),
        )
    return report
