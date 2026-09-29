"""Pure-Torch reference and error metrics for the SM90 push FP8 tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reference_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run a complete MoE forward with all experts resident on one device."""
    num_tokens, hidden = hidden_states.shape
    num_experts, two_intermediate, weight_hidden = w13.shape
    assert weight_hidden == hidden and two_intermediate % 2 == 0
    intermediate = two_intermediate // 2
    assert tuple(w2.shape) == (num_experts, hidden, intermediate)

    x = hidden_states.to(compute_dtype)
    w13 = w13.to(compute_dtype)
    w2 = w2.to(compute_dtype)
    output = torch.zeros(
        num_tokens,
        hidden,
        dtype=compute_dtype,
        device=hidden_states.device,
    )
    for expert in range(num_experts):
        selected = topk_ids == expert
        if not bool(selected.any()):
            continue
        token_indices, slot_indices = selected.nonzero(as_tuple=True)
        fc1 = x[token_indices] @ w13[expert].T
        gate, up = fc1[:, :intermediate], fc1[:, intermediate:]
        fc2 = (F.silu(gate) * up) @ w2[expert].T
        route_weights = topk_weights[token_indices, slot_indices].unsqueeze(1)
        output.index_add_(0, token_indices, fc2 * route_weights.to(compute_dtype))
    return output


def quant_dequant_128x128(weight: torch.Tensor) -> torch.Tensor:
    """Quantize and dequantize one matrix with 128x128 FP8 block scales."""
    rows, columns = weight.shape
    assert rows % 128 == 0 and columns % 128 == 0
    blocks = weight.float().reshape(rows // 128, 128, columns // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3))
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    quantized = (
        (blocks / scale[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return (quantized.float() * scale[:, None, :, None]).reshape(rows, columns)


def reference_moe_fp8_weights_streaming(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Run the reference with FP8-block-dequantized weights one expert at a time."""
    num_tokens, hidden = hidden_states.shape
    num_experts, two_intermediate, weight_hidden = w13.shape
    assert weight_hidden == hidden and two_intermediate % 2 == 0
    intermediate = two_intermediate // 2
    assert tuple(w2.shape) == (num_experts, hidden, intermediate)

    x = hidden_states.float()
    output = torch.zeros(
        num_tokens, hidden, dtype=torch.float32, device=hidden_states.device
    )
    for expert in range(num_experts):
        selected = topk_ids == expert
        if not bool(selected.any()):
            continue
        token_indices, slot_indices = selected.nonzero(as_tuple=True)
        w13_dequant = quant_dequant_128x128(w13[expert])
        w2_dequant = quant_dequant_128x128(w2[expert])
        fc1 = x[token_indices] @ w13_dequant.T
        fc2 = (F.silu(fc1[:, :intermediate]) * fc1[:, intermediate:]) @ w2_dequant.T
        route_weights = topk_weights[token_indices, slot_indices].unsqueeze(1).float()
        output.index_add_(0, token_indices, fc2 * route_weights)
    return output


def dequant_act_1x128(
    activations_fp8: torch.Tensor, activation_scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize per-token, per-128-channel FP8 activations to FP32."""
    rows, columns = activations_fp8.shape
    assert columns % 128 == 0
    assert tuple(activation_scales.shape) == (rows, columns // 128)
    return (
        activations_fp8.float().reshape(rows, columns // 128, 128)
        * activation_scales.unsqueeze(-1)
    ).reshape(rows, columns)


def dequant_weight_128x128(
    weight_fp8: torch.Tensor, weight_scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize a matrix carrying one FP32 scale per 128x128 FP8 block."""
    rows, columns = weight_fp8.shape
    assert rows % 128 == 0 and columns % 128 == 0
    assert tuple(weight_scales.shape) == (rows // 128, columns // 128)
    return (
        weight_fp8.float().reshape(rows // 128, 128, columns // 128, 128)
        * weight_scales.reshape(rows // 128, 1, columns // 128, 1)
    ).reshape(rows, columns)


def compare(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    err_ratio: float = 0.15,
    cos_min: float = 0.98,
) -> dict[str, float | bool]:
    """Return normalized error metrics and their combined acceptance result."""
    output = output.float()
    reference = reference.float()
    absolute_error = (output - reference).abs()
    reference_rms = reference.square().mean().sqrt().clamp_min(1e-6)
    error_rms = (output - reference).square().mean().sqrt()
    cosine = F.cosine_similarity(output.flatten(), reference.flatten(), dim=0)
    metrics: dict[str, float | bool] = {
        "max_abs": absolute_error.max().item(),
        "mean_abs": absolute_error.mean().item(),
        "err_ratio": float(error_rms / reference_rms),
        "cos": float(cosine),
    }
    metrics["passed"] = bool(
        metrics["err_ratio"] < err_ratio and metrics["cos"] > cos_min
    )
    return metrics
