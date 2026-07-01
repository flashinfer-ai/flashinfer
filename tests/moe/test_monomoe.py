"""Accuracy test for the single-kernel block-FP8 MoE (monomoe).

The kernel is hard-specialized to the Qwen3.5-35B shape on Hopper (SM90a):
E=256 experts, N=512, K=2048, up to BS=8 tokens, block-wise (128x128) FP8
weights with per-token-dynamic 1x128 activation quantization.

The Python reference replicates the kernel's exact math (block-wise dequant
GEMM, SiLU gating, block-wise re-quant of the intermediate, block-wise down
GEMM) so the comparison is apples-to-apples in fp8.
"""

import pytest
import torch

from flashinfer.fused_moe import has_monomoe, mono_moe
from flashinfer.utils import is_sm90a_supported

# Fixed geometry of the only compiled variant.
E = 256
N = 512  # N_half: gate and up each have N rows
K = 2048
BLOCK = 128


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
            s = amax / 448.0
            scales[:, ri, ci] = s.reshape(Ee)
            w_fp8[:, r0:r1, c0:c1] = (block / s).clamp(-448, 448)
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
        scales[g] = amax / 448.0
        x_fp8[g0:g1] = (block * (448.0 / amax)).clamp(-448, 448).to(torch.float8_e4m3fn)
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


def _python_reference(x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids):
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


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("top_k", [1, 8])
def test_monomoe_accuracy(m, top_k):
    dev = torch.device("cuda")
    if not is_sm90a_supported(dev):
        pytest.skip("monomoe requires SM90a (Hopper)")
    if not has_monomoe():
        pytest.skip("monomoe extension unavailable (failed to build/load)")

    torch.manual_seed(42)
    # Up weights are stored [E, 2*N, K] = [gate(N rows) || up(N rows)].
    w13_f = torch.randn(E, 2 * N, K, device=dev) * 0.1
    w2_f = torch.randn(E, K, N, device=dev) * 0.1
    w13_fp8, s13 = _quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = _quant_fp8_block_wise(w2_f)

    x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(m, E, device=dev, dtype=torch.bfloat16)
    topk_w, topk_ids = _routing_softmax_topk(logits, top_k)

    ref = _python_reference(x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids)

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
    )

    assert out.shape == (m, K)
    assert out.dtype == torch.bfloat16

    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), ref.float().reshape(-1), dim=0
    ).item()
    max_abs = (out.float() - ref.float()).abs().max().item()
    print(f"\n[monomoe] m={m} top_k={top_k}: cos_sim={cos:.5f} max_abs={max_abs:.4f}")
    assert cos > 0.98, f"cosine similarity too low: {cos:.5f}"
