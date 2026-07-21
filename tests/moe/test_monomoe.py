"""Accuracy test for the single-kernel block-FP8 MoE (monomoe).

The kernel serves the fixed E=256/N=512/K=2048 shape on Hopper (SM90a):
block-wise (128x128) FP8 weights, per-token-dynamic 1x128 activation
quantization, up to 8 tokens (the BS8 kernel).

The block-FP8 Python reference (inlined below) replicates the kernel's exact
math (block-wise dequant GEMM, SiLU gating, block-wise re-quant of the
intermediate, block-wise down GEMM), so the comparison is apples-to-apples in
fp8 and measures kernel bugs rather than fp8-vs-fp32 drift.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fused_moe import alloc_scratchpad, has_monomoe, mono_moe
from flashinfer.utils import is_sm90a_supported

BLOCK = 128


# ── Block-FP8 quantization / reference helpers (inlined) ─────────────────────
# The block-FP8 Python reference replicates the kernel's exact math (block-wise
# dequant GEMM, SiLU gating, block-wise re-quant of the intermediate, block-wise
# down GEMM), so the accuracy comparison is apples-to-apples in fp8 and measures
# kernel bugs rather than fp8-vs-fp32 drift.  Pure PyTorch, single shape family.
_FP8_MAX = 448.0  # e4m3 dynamic range


def _quant_fp8_block_wise(w, block_row=128, block_col=128):
    """Block-wise FP8 quantization. w: [E, rows, cols] -> (fp8, scales)."""
    Ee, rows, cols = w.shape
    rb = (rows + block_row - 1) // block_row
    cb = (cols + block_col - 1) // block_col
    wf = w.float()
    scales = torch.zeros(Ee, rb, cb, device=w.device, dtype=torch.float32)
    w_fp8 = torch.zeros_like(wf)
    for ri in range(rb):
        r0, r1 = ri * block_row, min((ri + 1) * block_row, rows)
        for ci in range(cb):
            c0, c1 = ci * block_col, min((ci + 1) * block_col, cols)
            block = wf[:, r0:r1, c0:c1]
            amax = block.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
            s = amax / _FP8_MAX
            scales[:, ri, ci] = s.reshape(Ee)
            w_fp8[:, r0:r1, c0:c1] = (block / s).clamp(-_FP8_MAX, _FP8_MAX)
    return w_fp8.to(torch.float8_e4m3fn), scales


def _quant_act_block_wise(x_float, group_size=128):
    """Per-token-dynamic 1xgroup_size activation quant. x_float: [K]."""
    n_groups = x_float.shape[0] // group_size
    x_fp8 = torch.zeros_like(x_float, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(n_groups, device=x_float.device, dtype=torch.float32)
    for g in range(n_groups):
        g0, g1 = g * group_size, (g + 1) * group_size
        block = x_float[g0:g1]
        amax = block.abs().max().clamp(min=1e-12)
        scales[g] = amax / _FP8_MAX
        x_fp8[g0:g1] = (
            (block * (_FP8_MAX / amax))
            .clamp(-_FP8_MAX, _FP8_MAX)
            .to(torch.float8_e4m3fn)
        )
    return x_fp8, scales


def _block_wise_gemm(w_fp8, scales_bw, x_fp8, x_scales, block_row=128, block_col=128):
    """result[row] = sum_blocks W*x * w_scale * x_scale.  w:[rows,cols], x:[cols]."""
    rows, cols = w_fp8.shape
    wf, xf = w_fp8.float(), x_fp8.float()
    rb, cb = scales_bw.shape
    result = torch.zeros(rows, device=w_fp8.device, dtype=torch.float32)
    for ri in range(rb):
        r0, r1 = ri * block_row, min((ri + 1) * block_row, rows)
        for ci in range(cb):
            c0, c1 = ci * block_col, min((ci + 1) * block_col, cols)
            result[r0:r1] += (
                (wf[r0:r1, c0:c1] @ xf[c0:c1]) * scales_bw[ri, ci] * x_scales[ci]
            )
    return result


def _routing_softmax_topk(logits, top_k):
    """Greedy softmax->topk->renormalize, lowest-index tie-break (matches kernel)."""
    scores = torch.softmax(logits.float(), dim=-1)
    M = scores.shape[0]
    ids = torch.zeros(M, top_k, dtype=torch.int64, device=logits.device)
    wts = torch.zeros(M, top_k, dtype=torch.float32, device=logits.device)
    s = scores.clone()
    for k in range(top_k):
        v, idx = s.max(dim=-1)
        wts[:, k] = v
        ids[:, k] = idx
        s.scatter_(1, idx.unsqueeze(1), float("-inf"))
    wts = wts / wts.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return wts, ids


def _python_reference_fp8(x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids, N, K):
    """Block-wise fp8 reference matching the kernel's math.  Returns out [M, K]."""
    M, top_k = x.shape[0], topk_ids.shape[1]
    out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    for tok in range(M):
        xq, x_scales = _quant_act_block_wise(x[tok].float(), group_size=128)
        for ki in range(top_k):
            eid = int(topk_ids[tok, ki])
            rw = float(topk_w[tok, ki])
            raw = _block_wise_gemm(w13_fp8[eid], s13[eid], xq, x_scales)
            gate, up = raw[:N], raw[N:]
            silu = rw * (up * gate) / (1.0 + torch.exp(-gate))
            silu_bf16 = silu.bfloat16()
            sq, s2_act = _quant_act_block_wise(silu_bf16.float(), group_size=128)
            out[tok] += _block_wise_gemm(w2_fp8[eid], s2[eid], sq, s2_act).bfloat16()
    return out


def _cosine(a, b):
    """Flattened cosine similarity of two tensors."""
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


# (E, N, K) test shapes.  e256_n512_k2048: has a BS16 companion;
# e64_n512_k2048: BS8-only, cheap.
SHAPES = [(256, 512, 2048), (64, 512, 2048)]


def _make_weights(dev, E, N, K, scale=0.1, seed=42):
    """Quantized block-FP8 up/down weights (fp8 tensors + scales) for a shape."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w13_f = torch.randn(E, 2 * N, K, device=dev, generator=g) * scale
    w2_f = torch.randn(E, K, N, device=dev, generator=g) * scale
    w13_fp8, s13 = _quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = _quant_fp8_block_wise(w2_f)
    return w13_fp8, s13, w2_fp8, s2


def _run_and_compare(x, logits, weights, N, K, top_k, scratchpad=None):
    """Run mono_moe against the block-FP8 reference; return (out, cos)."""
    w13_fp8, s13, w2_fp8, s2 = weights
    topk_w, topk_ids = _routing_softmax_topk(logits, top_k)
    ref = _python_reference_fp8(x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids, N, K)

    # mono_moe applies the up-weight TMA interleave internally by default.
    out = mono_moe(
        x,
        logits,
        w13_fp8,
        s13,
        w2_fp8,
        s2,
        top_k=top_k,
        scoring_func="softmax",
        renormalize=True,
        scratchpad=scratchpad,
    )
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16
    cos = _cosine(out, ref)
    return out, cos


def _require_monomoe(dev):
    if not is_sm90a_supported(dev):
        pytest.skip("monomoe requires SM90a (Hopper)")
    if not has_monomoe():
        pytest.skip("monomoe unavailable for the E=256/N=512/K=2048 shape")


# Single fixed shape (E=256, N=512, K=2048), BS8 kernel: M <= 8 tokens.
@pytest.mark.parametrize("m", [1, 2, 8])
@pytest.mark.parametrize("top_k", [1, 8])
def test_monomoe_accuracy(m, top_k):
    E, N, K = 256, 512, 2048
    dev = torch.device("cuda")
    _require_monomoe(dev)

    torch.manual_seed(42)
    weights = _make_weights(dev, E, N, K)
    x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(m, E, device=dev, dtype=torch.bfloat16)

    _, cos = _run_and_compare(x, logits, weights, N, K, top_k)
    print(f"\n[monomoe] E={E} N={N} K={K} m={m} top_k={top_k}: cos_sim={cos:.5f}")
    assert cos > 0.98, f"cosine similarity too low: {cos:.5f}"


@pytest.mark.parametrize("m", [1, 8])
def test_monomoe_scratchpad_reuse_no_contamination(m):
    """A scratchpad reused across calls with DIFFERENT inputs must not leak
    state between launches.

    The kernel's cross-block handoffs (the act-scale sentinel and the
    Phase-4->5 arrival counters) live in the scratchpad and are meant to be
    self-maintaining via a per-launch parity double-buffer.  If that reset
    discipline were wrong, a second launch on the same buffer could observe
    a stale scale/flag from the first and corrupt its output.  We run several
    independent problems through ONE scratchpad and require each to match its
    own reference — a fresh-scratchpad control run is compared against too.
    """
    E, N, K = 256, 512, 2048
    dev = torch.device("cuda")
    _require_monomoe(dev)

    weights = _make_weights(dev, E, N, K, seed=7)
    scratch = alloc_scratchpad(dev)

    # Distinct (x, logits) per iteration so a leaked buffer would show up as a
    # mismatch against THIS iteration's reference.
    prev_out = None
    for i in range(4):
        torch.manual_seed(100 + i)
        x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
        logits = torch.randn(m, E, device=dev, dtype=torch.bfloat16)

        out_reuse, cos_reuse = _run_and_compare(
            x, logits, weights, N, K, top_k=8, scratchpad=scratch
        )
        # Control: a fresh scratchpad for the same inputs must match exactly.
        out_fresh, _ = _run_and_compare(
            x, logits, weights, N, K, top_k=8, scratchpad=alloc_scratchpad(dev)
        )

        print(f"\n[monomoe reuse] m={m} iter={i}: cos_sim={cos_reuse:.5f}")
        assert cos_reuse > 0.98, f"reuse iter {i}: cos too low {cos_reuse:.5f}"
        # Reused-buffer output must match the fresh-buffer output for the same
        # inputs.  Not bit-exact: the Phase-4 cross-block atomicAdd reduces in
        # nondeterministic order, so runs differ by up to a rounding ULP —
        # contamination (a leaked scale/flag) would instead flip whole
        # rows/experts, far above this tolerance.
        cos_rf = _cosine(out_reuse, out_fresh)
        assert cos_rf > 0.9999, (
            f"reuse iter {i}: reuse vs fresh diverged ({cos_rf:.6f})"
        )
        if prev_out is not None:
            # Different inputs must produce a different output (guards against
            # the kernel silently returning a stale buffer).
            assert not torch.equal(out_reuse, prev_out)
        prev_out = out_reuse.clone()


@pytest.mark.parametrize("scale", [1e-2, 3e-3])
def test_monomoe_small_scales(scale):
    """Correctness must hold when weight/activation magnitudes are tiny.

    Block-FP8 rescales each 128x128 tile to the e4m3 range, so uniformly
    scaling the inputs is numerically inert *until* the OUTPUT magnitude
    (~scale^2 here) sinks below the fp8 pipeline's dynamic-range floor — at
    which point the fp8 reference itself loses correlation with fp32, so the
    kernel can no longer be distinguished from it.  These scales keep the
    output above that floor (~1e-6) while still exercising the tiny-scale
    path: the act-scale sentinel publish clamps to >= FLT_MIN so a
    legitimately small scale can never flush to the 0.0f "not published"
    sentinel and mis-handoff a consumer.
    """
    E, N, K = 256, 512, 2048
    dev = torch.device("cuda")
    _require_monomoe(dev)

    torch.manual_seed(2024)
    weights = _make_weights(dev, E, N, K, scale=scale, seed=11)
    x = torch.randn(8, K, device=dev, dtype=torch.bfloat16) * scale
    logits = torch.randn(8, E, device=dev, dtype=torch.bfloat16)

    out, cos = _run_and_compare(x, logits, weights, N, K, top_k=8)
    # No NaN/Inf even at the fp8 subnormal boundary.
    assert torch.isfinite(out.float()).all(), "small-scale output has NaN/Inf"
    print(f"\n[monomoe small-scale] scale={scale:g}: cos_sim={cos:.5f}")
    assert cos > 0.98, f"cosine similarity too low: {cos:.5f}"
