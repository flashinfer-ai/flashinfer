"""End-to-end W4A8 MXFP4 MoE on Hopper (SM90), correctness-first path.

Validates that the from-scratch CuTe DSL W4A8 MXFP4 grouped GEMM
(`w4a8_mxfp4_grouped_gemm`) slots correctly into a real MoE routing layer. The
routing / gather / scatter+finalize is done here in torch (the reference), so this
establishes the golden end-to-end result that the future *fused* kernel path
(cp.async per-row gather + epilogue scatter, mirroring the Blackwell CuTe DSL MoE)
will be checked against.

Pipeline (single grouped GEMM = MoE "GEMM1" without SwiGLU/GEMM2):
  1. Router assigns each token to top_k experts (random) with routing weights.
  2. Sort the (token, slot) pairs by expert -> per-expert gathered FP8 activations.
  3. `w4a8_mxfp4_grouped_gemm` computes, per expert, C = A @ dequant(W_mxfp4, scale)^T.
  4. Scatter rows back to tokens, weighted-sum over top_k -> Y [T, N].
Reference recomputes the same with exact MXFP4 LUT dequant; because both FP4 LUT
values and the UE8M0 power-of-2 scale are exactly representable in FP8 e4m3 (no
overflow in the tested range) and the activations are already FP8, the kernel is
expected to match the reference to GEMM accumulation / FP16-output precision.
"""

import os
import sys

import pytest
import torch

import cutlass

# MXFP4 (FP4 e2m1) code -> value LUT and UE8M0 baseline, matching the kernel.
_FP4_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
_SCALE_BASE = 127


def _import_kernel():
    """Import the callable from the kernel module (path-insensitive)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from flashinfer.fused_moe.cute_dsl.w4a8_mxfp4_grouped_gemm_sm90 import (
        w4a8_mxfp4_grouped_gemm,
    )

    return w4a8_mxfp4_grouped_gemm


def _make_expert_weight(n, k, device, seed):
    """Return (packed_u8 [N, K//2], scale_u8 [N, K//32], w_fp32 [N, K]).

    Packs random FP4 codes 2 nibbles/byte (low nibble = even K index) and picks
    UE8M0 scale bytes in a small range around the baseline (no e4m3 overflow), so
    the FP8 dequant is exact and the reference matches the kernel tightly.
    """
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)  # [N, K]
    flat = codes.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).reshape(n, k // 2).contiguous()
    scale = torch.randint(
        _SCALE_BASE - 1, _SCALE_BASE + 3, (n, k // 32), generator=g, dtype=torch.uint8
    )
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32)
    vals = lut[codes.long()]  # [N, K] fp32
    se = (scale.to(torch.int32) - _SCALE_BASE).float()  # [N, K//32]
    w_fp32 = vals * (2.0**se).repeat_interleave(32, dim=1)  # [N, K]
    return (
        packed.to(device),
        scale.to(device),
        w_fp32.to(device),
    )


def _reference_moe(x_fp8, topk_ids, topk_w, expert_w_fp32):
    """Golden MoE: Y[t] = sum_slot w[t,slot] * (X_f32[t] @ W_{ids[t,slot]}^T)."""
    t, k = x_fp8.shape
    n = expert_w_fp32[0].shape[0]
    x_f32 = x_fp8.float()
    y = torch.zeros(t, n, dtype=torch.float32, device=x_fp8.device)
    top_k = topk_ids.shape[1]
    for slot in range(top_k):
        for e, w in enumerate(expert_w_fp32):
            mask = topk_ids[:, slot] == e
            if mask.any():
                y[mask] += topk_w[mask, slot, None] * (x_f32[mask] @ w.t())
    return y


_OUT_DTYPES = {
    "float16": (torch.float16, cutlass.Float16),
    "bfloat16": (torch.bfloat16, cutlass.BFloat16),
}


def run_moe(
    num_tokens=128, num_experts=8, top_k=2, k=256, n=256, seed=0, out_dtype="float16"
):
    device = torch.device("cuda")
    torch_c_dtype, cutlass_c_dtype = _OUT_DTYPES[out_dtype]
    w4a8_mxfp4_grouped_gemm = _import_kernel()
    torch.manual_seed(seed)

    # Activations: FP8 e4m3, scaled so MoE outputs stay O(1) for a tight tolerance.
    x_f32 = torch.randn(num_tokens, k, device=device) / (k**0.5)
    x_fp8 = x_f32.to(torch.float8_e4m3fn)

    # Expert MXFP4 weights + UE8M0 scales.
    packed_w, scale_w, w_fp32 = [], [], []
    for e in range(num_experts):
        p, s, w = _make_expert_weight(n, k, device, seed=seed * 1000 + e)
        packed_w.append(p)
        scale_w.append(s)
        w_fp32.append(w)

    # Router: each token picks top_k DISTINCT experts, random positive weights.
    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.long, device=device)
    for t in range(num_tokens):
        topk_ids[t] = torch.randperm(num_experts, device=device)[:top_k]
    topk_w = torch.rand(num_tokens, top_k, device=device) + 0.5

    # ---- Routing / gather: sort (token, slot) pairs by expert. ----
    tok_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k)
    flat_tok = tok_idx.reshape(-1)  # expanded -> token
    flat_slot = (
        torch.arange(top_k, device=device)
        .unsqueeze(0)
        .expand(num_tokens, -1)
        .reshape(-1)
    )
    flat_exp = topk_ids.reshape(-1)  # expanded -> expert

    a_list, b_list, s_list, c_list = [], [], [], []
    problem_sizes = []
    routed = []  # per-group: (token_idx_tensor, slot_idx_tensor) for scatter
    for e in range(num_experts):
        sel = (flat_exp == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue  # skip empty experts (grouped GEMM needs M_g >= 1)
        toks = flat_tok[sel]
        slots = flat_slot[sel]
        a_g = x_fp8[toks].contiguous()  # [M_e, K] FP8
        c_g = torch.empty(toks.numel(), n, dtype=torch_c_dtype, device=device)
        a_list.append(a_g)
        b_list.append(packed_w[e])
        s_list.append(scale_w[e])
        c_list.append(c_g)
        problem_sizes.append((toks.numel(), n, k, 1))
        routed.append((toks, slots))

    w4a8_mxfp4_grouped_gemm(
        a_list,
        b_list,
        s_list,
        c_list,
        problem_sizes,
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass_c_dtype,
    )
    torch.cuda.synchronize()

    # ---- Scatter + finalize: rows back to tokens, weighted sum over top_k. ----
    y = torch.zeros(num_tokens, n, dtype=torch.float32, device=device)
    for (toks, slots), c_g in zip(routed, c_list, strict=False):
        w = topk_w[toks, slots].unsqueeze(1)
        y.index_add_(0, toks, w * c_g.float())

    y_ref = _reference_moe(x_fp8, topk_ids, topk_w, w_fp32)

    diff = (y - y_ref).abs()
    denom = y_ref.abs().clamp_min(1e-3)
    # BF16 output (~2^-8 mantissa) rounds each expert's row before the top_k
    # fp32 accumulation, so it needs a looser tolerance than FP16.
    rel_tol, abs_tol = (0.02, 5e-3) if out_dtype == "float16" else (0.04, 2e-2)
    frac_bad = (diff > (denom * rel_tol + abs_tol)).float().mean().item()
    print(
        f"tokens={num_tokens} experts={num_experts} top_k={top_k} K={k} N={n} "
        f"out={out_dtype} | max_abs={diff.max().item():.4f} frac_bad={frac_bad:.4%} "
        f"|y|={y.abs().mean().item():.4f} |ref|={y_ref.abs().mean().item():.4f}"
    )
    return y, y_ref, frac_bad


def _sm90():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


@pytest.mark.skipif(not _sm90(), reason="W4A8 MXFP4 grouped GEMM requires Hopper SM90")
@pytest.mark.parametrize("out_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("num_tokens", [64, 128])
def test_w4a8_mxfp4_moe_e2e(num_tokens, top_k, out_dtype):
    _, _, frac_bad = run_moe(num_tokens=num_tokens, top_k=top_k, out_dtype=out_dtype)
    # BF16 rounds each expert row before the top_k fp32 accumulation, so a tiny
    # fraction of elements land just over the FP16 threshold -- allow a bit more.
    assert frac_bad < (1e-3 if out_dtype == "float16" else 3e-3)


def run_moe_fused_scatter(num_tokens=128, num_experts=8, top_k=2, k=256, n=256, seed=0):
    """FS-1 fused token scatter: the kernel scatter-ADDS each expert's output rows
    into a shared FP32 MoE output via a per-group route map (local row -> token)
    + routing weight, with atomicAdd so top_k>1 experts accumulate per token. No
    torch scatter -- this is the fused finalize.
    """
    device = torch.device("cuda")
    w4a8_mxfp4_grouped_gemm = _import_kernel()
    torch.manual_seed(seed)

    x_f32 = torch.randn(num_tokens, k, device=device) / (k**0.5)
    x_fp8 = x_f32.to(torch.float8_e4m3fn)

    packed_w, scale_w, w_fp32 = [], [], []
    for e in range(num_experts):
        p, s, w = _make_expert_weight(n, k, device, seed=seed * 1000 + e)
        packed_w.append(p)
        scale_w.append(s)
        w_fp32.append(w)

    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.long, device=device)
    for t in range(num_tokens):
        topk_ids[t] = torch.randperm(num_experts, device=device)[:top_k]
    topk_w = torch.rand(num_tokens, top_k, device=device) + 0.5

    flat_tok = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    flat_slot = (
        torch.arange(top_k, device=device)
        .unsqueeze(0)
        .expand(num_tokens, -1)
        .reshape(-1)
    )
    flat_exp = topk_ids.reshape(-1)

    a_list, b_list, s_list, route_list, wt_list = [], [], [], [], []
    problem_sizes = []
    for e in range(num_experts):
        sel = (flat_exp == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        toks = flat_tok[sel]
        slots = flat_slot[sel]
        a_list.append(x_fp8[toks].contiguous())
        b_list.append(packed_w[e])
        s_list.append(scale_w[e])
        route_list.append(toks.to(torch.int32).contiguous())  # local row -> token
        wt_list.append(topk_w[toks, slots].to(torch.float32).contiguous())
        problem_sizes.append((toks.numel(), n, k, 1))

    # Shared FP32 output (atomicAdd buffer), zeroed.
    output = torch.zeros(num_tokens, n, dtype=torch.float32, device=device)
    w4a8_mxfp4_grouped_gemm(
        a_list,
        b_list,
        s_list,
        None,
        problem_sizes,
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass.Float16,
        route_maps=route_list,
        weights=wt_list,
        output=output,
        # top_k==1 -> each token written once -> plain store (no atomicAdd needed).
        no_accumulate=(top_k == 1),
    )
    torch.cuda.synchronize()

    y_ref = _reference_moe(x_fp8, topk_ids, topk_w, w_fp32)

    diff = (output - y_ref).abs()
    denom = y_ref.abs().clamp_min(1e-3)
    frac_bad = (diff > (denom * 0.02 + 5e-3)).float().mean().item()
    print(
        f"[fused-scatter top_k={top_k}] tokens={num_tokens} experts={num_experts} "
        f"K={k} N={n} | max_abs={diff.max().item():.4f} frac_bad={frac_bad:.4%} "
        f"|y|={output.abs().mean().item():.4f} |ref|={y_ref.abs().mean().item():.4f}"
    )
    return output, y_ref, frac_bad


@pytest.mark.skipif(not _sm90(), reason="W4A8 MXFP4 grouped GEMM requires Hopper SM90")
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("num_tokens", [64, 128])
def test_w4a8_mxfp4_moe_fused_scatter(num_tokens, top_k):
    _, _, frac_bad = run_moe_fused_scatter(num_tokens=num_tokens, top_k=top_k)
    assert frac_bad < 1e-3


def run_moe_fused_gather(num_tokens=128, num_experts=8, top_k=2, k=256, n=256, seed=0):
    """Fused gather: the kernel reads A rows from a shared activation tensor via a
    per-group route map (cp.async gather) instead of pre-gathered per-group A. The
    output stays per-group C (torch scatter separate); this isolates the gather.
    """
    device = torch.device("cuda")
    w4a8_mxfp4_grouped_gemm = _import_kernel()
    torch.manual_seed(seed)

    x_f32 = torch.randn(num_tokens, k, device=device) / (k**0.5)
    x_fp8 = x_f32.to(torch.float8_e4m3fn)

    packed_w, scale_w, w_fp32 = [], [], []
    for e in range(num_experts):
        p, s, w = _make_expert_weight(n, k, device, seed=seed * 1000 + e)
        packed_w.append(p)
        scale_w.append(s)
        w_fp32.append(w)

    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.long, device=device)
    for t in range(num_tokens):
        topk_ids[t] = torch.randperm(num_experts, device=device)[:top_k]
    flat_tok = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    flat_exp = topk_ids.reshape(-1)

    b_list, s_list, c_list, route_list, problem_sizes, routed = [], [], [], [], [], []
    for e in range(num_experts):
        sel = (flat_exp == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        toks = flat_tok[sel]
        b_list.append(packed_w[e])
        s_list.append(scale_w[e])
        route_list.append(toks.to(torch.int32).contiguous())  # local row -> X row
        c_list.append(torch.empty(toks.numel(), n, dtype=torch.float16, device=device))
        problem_sizes.append((toks.numel(), n, k, 1))
        routed.append((e, toks))

    w4a8_mxfp4_grouped_gemm(
        None,
        b_list,
        s_list,
        c_list,
        problem_sizes,
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass.Float16,
        activations=x_fp8,
        gather_route_maps=route_list,
    )
    torch.cuda.synchronize()

    x_ref = x_fp8.float()
    max_frac = 0.0
    for (e, toks), c_g in zip(routed, c_list, strict=False):
        c_ref = x_ref[toks] @ w_fp32[e].t()
        diff = (c_g.float() - c_ref).abs()
        denom = c_ref.abs().clamp_min(1e-3)
        max_frac = max(max_frac, (diff > (denom * 0.02 + 5e-3)).float().mean().item())
    print(
        f"[fused-gather top_k={top_k}] tokens={num_tokens} experts={num_experts} "
        f"K={k} N={n} | groups={len(c_list)} max_frac_bad={max_frac:.4%}"
    )
    return max_frac


@pytest.mark.skipif(not _sm90(), reason="W4A8 MXFP4 grouped GEMM requires Hopper SM90")
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("num_tokens", [64, 128])
def test_w4a8_mxfp4_moe_fused_gather(num_tokens, top_k):
    frac_bad = run_moe_fused_gather(num_tokens=num_tokens, top_k=top_k)
    assert frac_bad < 1e-3


if __name__ == "__main__":
    y, y_ref, frac_bad = run_moe()
    print("PASS (torch-scatter)" if frac_bad < 1e-3 else "FAIL")
    for tk in (1, 2):
        _, _, fb2 = run_moe_fused_scatter(top_k=tk)
        print(f"PASS (fused-scatter top_k={tk})" if fb2 < 1e-3 else f"FAIL ({tk})")
    for tk in (1, 2):
        fb3 = run_moe_fused_gather(top_k=tk)
        print(f"PASS (fused-gather top_k={tk})" if fb3 < 1e-3 else f"FAIL ({tk})")
