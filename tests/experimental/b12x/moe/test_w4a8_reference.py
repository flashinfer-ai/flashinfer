"""Tests for the w4a8 (MXFP8 x FP4) MoE reference oracle.

Pure-Torch gates that run before any kernel exists: decomposition
correctness, trace/full-oracle consistency, stage exactness on
exactly-representable inputs, and the raison-d'etre accuracy A/B
(w4a8 must beat w4a4 on matched inputs at the oracle level).
"""

from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics import (
    _fp4_encode_nibbles,
    fp4_quantize_values_torch,
    quant_dequant_mxfp8_torch,
)
from b12x.moe._shared.kernels.reference import (
    decompose_nvfp4_scales_to_mx_residual,
    moe_reference_f32,
    moe_reference_w4a8_mx,
    nvfp4_mx_residual_quality_report,
    trace_moe_reference_w4a8_route,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pack_fp4_rows(values: torch.Tensor) -> torch.Tensor:
    """Pack FP4-grid values [rows, cols] into [rows, cols/2] uint8 (lo nibble first)."""
    nibbles = _fp4_encode_nibbles(values)
    pair = nibbles.view(values.shape[0], values.shape[1] // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).contiguous()


def _quantize_weight_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [rows, K] f32 weights to NVFP4 with global scale 1.0.

    Returns (packed_u8 [rows, K/2], scales_e4m3 [rows, K/16] f32, dequant f32).
    """
    rows, cols = w.shape
    blocked = w.view(rows, cols // 16, 16)
    bmax = blocked.abs().amax(dim=-1, keepdim=True)
    scale = (bmax / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)
    q = fp4_quantize_values_torch((blocked / scale.clamp(min=1e-30)).view(rows, cols))
    packed = _pack_fp4_rows(q)
    dequant = (q.view(rows, cols // 16, 16) * scale).view(rows, cols)
    return packed, scale.squeeze(-1), dequant


def _quantize_weight_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [rows, K] f32 weights to MXFP4 (e8m0/K32, ceil scale).

    Returns (packed_u8, scale_bytes [rows, K/32] uint8, dequant f32).
    """
    rows, cols = w.shape
    blocked = w.view(rows, cols // 32, 32)
    bmax = blocked.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(bmax > 0, bmax / 6.0, torch.ones_like(bmax))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    byte = torch.where(bmax > 0, exponent + 127, torch.zeros_like(exponent)).to(torch.uint8)
    scale = torch.where(bmax > 0, torch.exp2(exponent), torch.zeros_like(exponent))
    q = fp4_quantize_values_torch(
        torch.where(scale > 0, blocked / scale.clamp(min=1e-30), torch.zeros_like(blocked)).view(rows, cols)
    )
    packed = _pack_fp4_rows(q)
    dequant = (q.view(rows, cols // 32, 32) * scale).view(rows, cols)
    return packed, byte.squeeze(-1), dequant


def test_decompose_roundtrip_exact_for_representable_residuals() -> None:
    torch.manual_seed(0)
    rows, kb = 64, 16
    # E4M3-representable scales with moderate pair ratios: residuals stay normal.
    raw = (torch.rand(rows, kb) * 4.0 + 0.25).to(torch.float8_e4m3fn).to(torch.float32)
    ue8m0, residual = decompose_nvfp4_scales_to_mx_residual(raw)
    assert ue8m0.shape == (rows, kb // 2)
    assert residual.shape == (rows, kb)
    recon = residual.to(torch.float32).view(rows, kb // 2, 2) * torch.exp2(
        ue8m0.to(torch.float32) - 127.0
    ).unsqueeze(-1)
    torch.testing.assert_close(recon.view(rows, kb), raw, atol=0, rtol=0)
    # Max-exponent residual of each pair must be in [1, 2).
    r = residual.to(torch.float32).view(rows, kb // 2, 2)
    rmax = r.amax(dim=-1)
    assert bool((rmax >= 1.0).all().item()) and bool((rmax < 2.0).all().item())


def test_decompose_edge_cases() -> None:
    scales = torch.tensor(
        [
            [448.0, 2.0**-9, 1.0, 1.0, 0.0, 3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    ue8m0, residual = decompose_nvfp4_scales_to_mx_residual(scales)
    r = residual.to(torch.float32)
    # Pair (448, 2^-9): shared exponent 8, partner residual 2^-17 flushes to 0.
    assert int(ue8m0[0, 0].item()) == 127 + 8
    assert r[0, 0].item() == 1.75  # 448 / 256
    assert r[0, 1].item() == 0.0
    # Pair (1, 1): exponent 0, residuals exactly 1.
    assert int(ue8m0[0, 1].item()) == 127
    assert r[0, 2].item() == 1.0 and r[0, 3].item() == 1.0
    # Pair (0, 3): zero scale contributes zero residual, exponent from partner.
    assert int(ue8m0[0, 2].item()) == 127 + 1
    assert r[0, 4].item() == 0.0 and r[0, 5].item() == 1.5
    # Pair (0, 0): neutral exponent byte, zero residuals.
    assert int(ue8m0[0, 3].item()) == 127
    assert r[0, 6].item() == 0.0 and r[0, 7].item() == 0.0

    report = nvfp4_mx_residual_quality_report(scales)
    assert report["flushed_fraction"] > 0.0
    assert report["max_pair_exponent_delta"] >= 17.0


def test_quality_report_clean_for_benign_scales() -> None:
    torch.manual_seed(1)
    raw = (torch.rand(32, 32) * 2.0 + 0.5).to(torch.float8_e4m3fn).to(torch.float32)
    report = nvfp4_mx_residual_quality_report(raw)
    assert report["flushed_fraction"] == 0.0
    assert report["max_rel_residual_error"] == 0.0  # residual mantissa == scale mantissa


def _make_synthetic_case(E: int, m: int, K: int, I_tp: int, top_k: int, seed: int):
    torch.manual_seed(seed)
    device = _DEVICE
    x = torch.randn(m, K, device=device) * 2.0
    w13 = torch.randn(E, 2 * I_tp, K, device=device) * 0.05
    w2 = torch.randn(E, K, I_tp, device=device) * 0.05
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:top_k] for _ in range(m)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(m, top_k, device=device), dim=-1)
    return x, w13, w2, topk_ids, topk_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_trace_matches_full_oracle() -> None:
    E, m, K, I_tp, top_k = 4, 5, 128, 64, 2
    x, w13, w2, topk_ids, topk_weights = _make_synthetic_case(E, m, K, I_tp, top_k, seed=2)

    w13_q = [_quantize_weight_nvfp4(w13[e]) for e in range(E)]
    w2_q = [_quantize_weight_nvfp4(w2[e]) for e in range(E)]
    w1_fp4 = torch.stack([q[0] for q in w13_q])
    w1_scales = torch.stack([q[1] for q in w13_q])
    w2_fp4 = torch.stack([q[0] for q in w2_q])
    w2_scales = torch.stack([q[1] for q in w2_q])
    w1_ue8m0, w1_res = decompose_nvfp4_scales_to_mx_residual(w1_scales)
    w2_ue8m0, w2_res = decompose_nvfp4_scales_to_mx_residual(w2_scales)
    ones = torch.ones(E, device=x.device)

    args = (
        x, w1_fp4, w1_ue8m0, w1_res, ones, w2_fp4, w2_ue8m0, w2_res, ones,
        topk_ids, topk_weights, E, K, I_tp,
    )
    activation_kwargs = {"swiglu_limit": 0.25}
    full = moe_reference_w4a8_mx(*args, **activation_kwargs)

    # Checkpoint W31 stores the up projection before the gate projection.
    # Swapping both the packed rows and their scale grids must preserve the
    # semantic W13 oracle output when the layout contract is supplied.
    def swap_gate_up_rows(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (tensor[:, I_tp:, ...], tensor[:, :I_tp, ...]), dim=1
        ).contiguous()

    args_w31 = (
        x,
        swap_gate_up_rows(w1_fp4),
        swap_gate_up_rows(w1_ue8m0),
        swap_gate_up_rows(w1_res),
        ones,
        w2_fp4,
        w2_ue8m0,
        w2_res,
        ones,
        topk_ids,
        topk_weights,
        E,
        K,
        I_tp,
    )
    full_w31 = moe_reference_w4a8_mx(
        *args_w31,
        w13_layout="w31",
        **activation_kwargs,
    )
    torch.testing.assert_close(full_w31, full, atol=0, rtol=0)

    accum = torch.zeros_like(full)
    for t in range(m):
        for r in range(top_k):
            trace = trace_moe_reference_w4a8_route(
                *args,
                token_idx=t,
                route_idx=r,
                **activation_kwargs,
            )
            accum[t] += trace.routed_out_accum
    torch.testing.assert_close(full, accum, atol=2e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w4a8_fc1_stage_exact_on_representable_inputs() -> None:
    """FC1 through the trace is exact when activations survive quant-dequant."""
    E, m, K, I_tp, top_k = 2, 3, 128, 64, 1
    device = _DEVICE
    torch.manual_seed(3)
    # Activations drawn from the E4M3 grid with block max exactly 448 -> the
    # MX quantizer's scale byte is 127 and the payload reproduces x exactly.
    x = (torch.randn(m, K, device=device) * 100).to(torch.float8_e4m3fn).to(torch.float32)
    x[:, ::32] = 448.0
    assert torch.equal(quant_dequant_mxfp8_torch(x), x)

    w13 = torch.randn(E, 2 * I_tp, K, device=device) * 0.05
    w2 = torch.randn(E, K, I_tp, device=device) * 0.05
    w13_q = [_quantize_weight_mxfp4(w13[e]) for e in range(E)]
    w2_q = [_quantize_weight_mxfp4(w2[e]) for e in range(E)]
    w1_fp4 = torch.stack([q[0] for q in w13_q])
    w1_bytes = torch.stack([q[1] for q in w13_q])
    w1_dequant = torch.stack([q[2] for q in w13_q])
    w2_fp4 = torch.stack([q[0] for q in w2_q])
    w2_bytes = torch.stack([q[1] for q in w2_q])

    topk_ids = torch.zeros(m, top_k, dtype=torch.int32, device=device)
    topk_ids[:, 0] = 1
    topk_weights = torch.ones(m, top_k, device=device)
    ones = torch.ones(E, device=device)

    trace = trace_moe_reference_w4a8_route(
        x, w1_fp4, w1_bytes, None, ones, w2_fp4, w2_bytes, None, ones,
        topk_ids, topk_weights, E, K, I_tp,
        token_idx=0, route_idx=0,
    )
    expected_up = w1_dequant[1, :I_tp] @ x[0]
    expected_gate = w1_dequant[1, I_tp:] @ x[0]
    torch.testing.assert_close(trace.up_out, expected_up, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(trace.gate_out, expected_gate, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(trace.x_dequant, x[0], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ["silu", "relu2"])
def test_w4a8_oracle_beats_w4a4_oracle(activation: str) -> None:
    """The accuracy raison-d'etre, at oracle level: cos(w4a8) >= cos(w4a4)."""
    E, m, K, I_tp, top_k = 8, 32, 256, 128, 4
    device = _DEVICE
    x, w13_full, w2_full, topk_ids, topk_weights = _make_synthetic_case(
        E, m, K, I_tp, top_k, seed=4
    )
    if activation == "relu2":
        w13_full = w13_full[:, :I_tp]

    w13_q = [_quantize_weight_nvfp4(w13_full[e]) for e in range(E)]
    w2_q = [_quantize_weight_nvfp4(w2_full[e]) for e in range(E)]
    w1_fp4 = torch.stack([q[0] for q in w13_q])
    w1_scales = torch.stack([q[1] for q in w13_q])
    w1_dequant = torch.stack([q[2] for q in w13_q])
    w2_fp4 = torch.stack([q[0] for q in w2_q])
    w2_scales = torch.stack([q[1] for q in w2_q])
    w2_dequant = torch.stack([q[2] for q in w2_q])
    ones = torch.ones(E, device=device)

    # Shared-weight pure-f32 reference (no activation quantization).
    is_gated = activation == "silu"
    ref = torch.zeros(m, K, device=device)
    for eid in range(E):
        mask = topk_ids == eid
        token_mask = mask.any(dim=1)
        if not bool(token_mask.any().item()):
            continue
        xs = x[token_mask].float()
        if is_gated:
            up = xs @ w1_dequant[eid, :I_tp].T
            gate = xs @ w1_dequant[eid, I_tp:].T
            inter = torch.sigmoid(gate) * gate * up
        else:
            inter = torch.square(torch.relu(xs @ w1_dequant[eid].T))
        down = inter @ w2_dequant[eid].T
        w = (topk_weights * mask.float()).sum(dim=1)[token_mask]
        ref[token_mask] += w.unsqueeze(1) * down

    # w4a8 oracle via the NVFP4 residual decomposition.
    w1_ue8m0, w1_res = decompose_nvfp4_scales_to_mx_residual(w1_scales)
    w2_ue8m0, w2_res = decompose_nvfp4_scales_to_mx_residual(w2_scales)
    out_w4a8 = moe_reference_w4a8_mx(
        x, w1_fp4, w1_ue8m0, w1_res, ones, w2_fp4, w2_ue8m0, w2_res, ones,
        topk_ids, topk_weights, E, K, I_tp, activation=activation,
    )

    # w4a4 oracle on the same packed weights/scales (global scales 1.0).
    from b12x._lib.intrinsics import swizzle_block_scale

    w1_swizzled = swizzle_block_scale(w1_scales.to(torch.float8_e4m3fn)).view(torch.uint8)
    w2_swizzled = swizzle_block_scale(w2_scales.to(torch.float8_e4m3fn)).view(torch.uint8)
    out_w4a4 = moe_reference_f32(
        x, w1_fp4, w1_swizzled, ones, w2_fp4, w2_swizzled, ones,
        ones, ones, topk_ids, topk_weights, E, K, I_tp, activation=activation,
    )

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(
            a.flatten().float(), b.flatten().float(), dim=0
        ).item()

    cos_w4a8 = _cos(out_w4a8, ref)
    cos_w4a4 = _cos(out_w4a4, ref)
    assert cos_w4a8 >= cos_w4a4, (cos_w4a8, cos_w4a4)
    assert cos_w4a8 > 0.995, cos_w4a8
