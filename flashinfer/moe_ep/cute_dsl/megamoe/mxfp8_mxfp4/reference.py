# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent Torch oracle for MXFP8-activation x MXFP4-weight MegaMoE.

This module intentionally has no dependency on ``kernel_src``, CUTLASS, CuTe
DSL, or another MegaMoE backend.  It defines the numerical function that the
SM100 CuTeDSL kernel implements.  Physical scale-factor swizzles and TMA tensor
layouts are outside this contract.

Logical MMA operand convention
------------------------------
For each expert GEMM the kernel-facing convention is::

    A = MXFP4 weights      (M, K, L), K-major
    B = MXFP8 activations  (N, K, L), K-major
    C = output             (M, N, L)

where ``L`` selects an expert.  The Torch-facing tensors below put ``L``
first, use row-major logical weight rows, and express the same operation as
``B @ A.T``.  Both A and B have one raw E8M0 scale byte per consecutive 32 K
elements.

Numerical boundaries
--------------------
* Input activations arrive already quantized to E4M3 plus E8M0/K32 and are
  dequantized to FP32 before FC1.
* MXFP4 weights are E2M1 nibble pairs plus E8M0/K32 and are dequantized to
  FP32 before each GEMM.
* FC1 GEMM accumulates in FP32.  Its output rows are gate/up interleaved in
  blocks of 32.  Optional clamping is applied in FP32, followed by
  ``up * (gate * sigmoid(gate))`` in FP32.
* The unweighted SwiGLU result is quantized directly from FP32 to E4M3 plus
  E8M0/K32 (no BF16 rounding first), then dequantized to FP32 for FC2.
* FC2 accumulates in FP32 and each route term is rounded once to BF16.
* Routing weights are applied after that BF16 route-term rounding.  Terms are
  multiplied by FP32 scores and accumulated in increasing top-k slot order in
  FP32.  The final result is rounded once to BF16.

Expert id ``-1`` is the sole masked-route sentinel.  Its per-route term is
zero and its score is ignored.  Any id below ``-1`` or at least the number of
experts is an input error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


BLOCK_SIZE = 32
E4M3_MAX = 448.0
E2M1_MAX = 6.0

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class MegaMoEReferenceResult:
    """Oracle outputs.

    ``route_terms`` has shape ``(rank, token, topk, hidden)`` and contains
    unweighted BF16 FC2 terms.  Masked routes contain zero.  ``output`` has
    shape ``(rank, token, hidden)`` and contains the post-routing BF16 result.
    """

    route_terms: torch.Tensor
    output: torch.Tensor


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.element_size() != 1:
        raise TypeError(
            f"expected a one-byte quantized tensor, got dtype={tensor.dtype}"
        )
    return tensor.contiguous().view(torch.uint8)


def decode_e8m0(scales: torch.Tensor) -> torch.Tensor:
    """Decode raw UE8M0 bytes as ``2 ** (byte - 127)`` in FP32.

    Byte ``0xff`` is the reserved NaN encoding and is rejected.  Byte zero is
    a valid, extremely small power-of-two scale; all-zero quantization blocks
    conventionally use it as well.
    """

    raw = _raw_bytes(scales)
    if bool((raw == 0xFF).any().item()):
        raise ValueError("E8M0 scale byte 0xff is reserved and not finite")
    exponent = raw.to(torch.int32) - 127
    return torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)


def _e8m0_scale_for_amax(amax: torch.Tensor, value_limit: float) -> torch.Tensor:
    """Encode ``ceil_pow2(amax / value_limit)`` as raw E8M0 bytes."""

    if not bool(torch.isfinite(amax).all().item()):
        raise ValueError("quantization input must contain only finite values")
    safe = torch.clamp(amax / value_limit, min=torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(safe))
    raw = torch.clamp(exponent + 127.0, min=0.0, max=254.0).to(torch.uint8)
    return torch.where(amax == 0, torch.zeros_like(raw), raw)


def dequantize_mxfp8(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize logical E4M3 data with trailing-dimension E8M0/K32 scales."""

    if data.ndim < 1 or data.shape[-1] % BLOCK_SIZE:
        raise ValueError("MXFP8 trailing dimension must be divisible by 32")
    expected = (*data.shape[:-1], data.shape[-1] // BLOCK_SIZE)
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"MXFP8 scales must have shape {expected}, got {tuple(scales.shape)}"
        )
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("Torch with float8_e4m3fn support is required")
    if data.dtype == torch.uint8:
        values = data.contiguous().view(fp8_dtype).float()
    elif data.dtype == fp8_dtype:
        values = data.float()
    else:
        raise TypeError(
            "MXFP8 data must be torch.float8_e4m3fn or its raw uint8 view, "
            f"got {data.dtype}"
        )
    expanded = decode_e8m0(scales).repeat_interleave(BLOCK_SIZE, dim=-1)
    return values * expanded


def quantize_mxfp8(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32-like values to E4M3 plus raw E8M0/K32.

    The scale is ``ceil_pow2(block_amax / 448)``.  The scaled E4M3 conversion
    uses Torch's round-to-nearest-even conversion.  No BF16 cast occurs.
    """

    if values.ndim < 1 or values.shape[-1] % BLOCK_SIZE:
        raise ValueError("MXFP8 trailing dimension must be divisible by 32")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("Torch with float8_e4m3fn support is required")
    fp32 = values.float()
    blocked = fp32.reshape(*fp32.shape[:-1], fp32.shape[-1] // BLOCK_SIZE, BLOCK_SIZE)
    raw_scale = _e8m0_scale_for_amax(blocked.abs().amax(dim=-1), E4M3_MAX)
    scale = decode_e8m0(raw_scale).repeat_interleave(BLOCK_SIZE, dim=-1)
    return (fp32 / scale).to(fp8_dtype), raw_scale


def dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize packed E2M1 weights with trailing-dimension E8M0/K32 scales."""

    if packed.ndim < 1:
        raise ValueError("MXFP4 data must have at least one dimension")
    logical_k = packed.shape[-1] * 2
    if logical_k % BLOCK_SIZE:
        raise ValueError("MXFP4 logical trailing dimension must be divisible by 32")
    expected = (*packed.shape[:-1], logical_k // BLOCK_SIZE)
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"MXFP4 scales must have shape {expected}, got {tuple(scales.shape)}"
        )
    raw = _raw_bytes(packed)
    codes = torch.empty(
        (*raw.shape[:-1], logical_k), dtype=torch.int64, device=raw.device
    )
    codes[..., 0::2] = (raw & 0x0F).to(torch.int64)
    codes[..., 1::2] = (raw >> 4).to(torch.int64)
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=raw.device)
    values = lut[codes]
    expanded = decode_e8m0(scales).repeat_interleave(BLOCK_SIZE, dim=-1)
    return values * expanded


def quantize_mxfp4(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-quantize rows to packed E2M1 plus raw E8M0/K32.

    E2M1 conversion is saturating round-to-nearest-even.  Even logical K is
    stored in the low nibble and odd logical K in the high nibble.
    """

    if values.ndim < 1 or values.shape[-1] % BLOCK_SIZE:
        raise ValueError("MXFP4 trailing dimension must be divisible by 32")
    fp32 = values.float()
    blocked = fp32.reshape(*fp32.shape[:-1], -1, BLOCK_SIZE)
    raw_scale = _e8m0_scale_for_amax(blocked.abs().amax(dim=-1), E2M1_MAX)
    scale = decode_e8m0(raw_scale).repeat_interleave(BLOCK_SIZE, dim=-1)
    normalized = fp32 / scale

    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=fp32.device)
    distance = (normalized.unsqueeze(-1) - lut).abs()
    minimum = distance.amin(dim=-1, keepdim=True)
    ties = distance == minimum
    # E2M1's mantissa bit is code bit zero.  RTNE chooses a code with that bit
    # clear at an exact midpoint.  ``argmax`` then selects the unique eligible
    # code (positive zero for an exact zero).
    even_code = (torch.arange(16, device=fp32.device, dtype=torch.int64) & 1) == 0
    prefer_even = ties & even_code
    candidates = torch.where(prefer_even.any(dim=-1, keepdim=True), prefer_even, ties)
    codes = candidates.to(torch.int64).argmax(dim=-1).to(torch.uint8)
    # Preserve the sign of a value that rounds to zero.  E2M1 has distinct
    # positive/negative-zero encodings and hardware RTNE preserves that sign.
    codes = torch.where(
        (codes == 0) & torch.signbit(normalized),
        torch.full_like(codes, 8),
        codes,
    )
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed, raw_scale


def interleave_gate_up_8(w13: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``gate || up`` rows to alternating 8-row blocks."""

    if w13.ndim < 2 or w13.shape[-2] % (2 * 8):
        raise ValueError("w13 output dimension must be divisible by 16")
    intermediate = w13.shape[-2] // 2
    gate, up = w13.split(intermediate, dim=-2)
    leading = gate.shape[:-2]
    hidden = gate.shape[-1]
    gate = gate.reshape(*leading, intermediate // 8, 8, hidden)
    up = up.reshape(*leading, intermediate // 8, 8, hidden)
    return torch.stack((gate, up), dim=-3).reshape(*leading, 2 * intermediate, hidden)


def _swiglu_interleaved_8(
    fc1: torch.Tensor, gate_up_clamp: Optional[float]
) -> torch.Tensor:
    rows, two_i = fc1.shape
    if two_i % (2 * 8):
        raise ValueError("FC1 output dimension must be divisible by 16")
    paired = fc1.reshape(rows, -1, 2, 8)
    gate = paired[:, :, 0, :]
    up = paired[:, :, 1, :]
    if gate_up_clamp is not None:
        limit = abs(float(gate_up_clamp))
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (up * (gate * torch.sigmoid(gate))).reshape(rows, two_i // 2)


def _validate_problem(
    input_q: torch.Tensor,
    input_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if input_q.ndim != 3:
        raise ValueError("input_q must have shape (rank, token, hidden)")
    ranks, tokens, hidden = input_q.shape
    if hidden % BLOCK_SIZE:
        raise ValueError("hidden must be divisible by 32")
    if tuple(input_scale.shape) != (ranks, tokens, hidden // BLOCK_SIZE):
        raise ValueError("input_scale must have shape (rank, token, hidden/32)")
    if topk_ids.ndim != 3 or tuple(topk_ids.shape[:2]) != (ranks, tokens):
        raise ValueError("topk_ids must have shape (rank, token, topk)")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must be int32 or int64")
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        raise ValueError("topk_weights must have the same shape as topk_ids")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must be float32")
    if not bool(torch.isfinite(topk_weights).all().item()):
        raise ValueError("topk_weights must be finite")
    if w13_packed.ndim != 3 or w2_packed.ndim != 3:
        raise ValueError("weights must have shapes (expert, output, packed_k)")
    experts, two_i, packed_h = w13_packed.shape
    if packed_h * 2 != hidden or two_i % (2 * BLOCK_SIZE):
        raise ValueError("w13_packed must have shape (expert, 2*I, hidden/2)")
    intermediate = two_i // 2
    if tuple(w2_packed.shape) != (experts, hidden, intermediate // 2):
        raise ValueError("w2_packed must have shape (expert, hidden, I/2)")
    if tuple(w13_scale.shape) != (experts, two_i, hidden // BLOCK_SIZE):
        raise ValueError("w13_scale must have shape (expert, 2*I, hidden/32)")
    if tuple(w2_scale.shape) != (experts, hidden, intermediate // BLOCK_SIZE):
        raise ValueError("w2_scale must have shape (expert, hidden, I/32)")
    tensors = (
        input_scale,
        topk_ids,
        topk_weights,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
    )
    if any(tensor.device != input_q.device for tensor in tensors):
        raise ValueError("all oracle tensors must be on the same device")
    invalid = (topk_ids < -1) | (topk_ids >= experts)
    if bool(invalid.any().item()):
        bad = int(topk_ids[invalid][0].item())
        raise ValueError(
            f"expert id {bad} is invalid; only -1 or [0, {experts}) are accepted"
        )
    return ranks, tokens, topk_ids.shape[-1], experts, hidden, intermediate


def compute_reference(
    *,
    input_q: torch.Tensor,
    input_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    gate_up_clamp: Optional[float] = None,
) -> MegaMoEReferenceResult:
    """Compute the independent MXFP8 x MXFP4 MegaMoE oracle.

    Shapes are:

    * ``input_q``: ``(R, T, H)`` E4M3; ``input_scale``: ``(R,T,H/32)``.
    * ``topk_ids`` and ``topk_weights``: ``(R, T, Ktop)``.
    * ``w13_packed``: ``(E, 2I, H/2)`` packed E2M1.  Its output rows use
      alternating 8-row gate/up blocks.  ``w13_scale`` is ``(E,2I,H/32)``.
    * ``w2_packed``: ``(E, H, I/2)`` packed E2M1; ``w2_scale`` is
      ``(E,H,I/32)``.

    In MMA notation, each weight tensor is logical A ``(M,K,L)`` and gathered
    tokens are logical B ``(N,K,L)``.  Packing halves only the stored K extent.
    """

    ranks, tokens, topk, experts, hidden, _intermediate = _validate_problem(
        input_q,
        input_scale,
        topk_ids,
        topk_weights,
        w13_packed,
        w13_scale,
        w2_packed,
        w2_scale,
    )
    input_fp32 = dequantize_mxfp8(input_q, input_scale)
    w13_fp32 = dequantize_mxfp4(w13_packed, w13_scale)
    w2_fp32 = dequantize_mxfp4(w2_packed, w2_scale)
    route_terms = torch.zeros(
        (ranks, tokens, topk, hidden),
        dtype=torch.bfloat16,
        device=input_q.device,
    )

    for expert in range(experts):
        routed = (topk_ids == expert).nonzero(as_tuple=False)
        if routed.numel() == 0:
            continue
        rank_idx, token_idx, slot_idx = routed.unbind(dim=1)
        gathered = input_fp32[rank_idx, token_idx]
        fc1 = gathered @ w13_fp32[expert].transpose(0, 1)
        swiglu = _swiglu_interleaved_8(fc1, gate_up_clamp)
        fc1_q, fc1_scale = quantize_mxfp8(swiglu)
        fc1_roundtrip = dequantize_mxfp8(fc1_q, fc1_scale)
        fc2 = fc1_roundtrip @ w2_fp32[expert].transpose(0, 1)
        route_terms[rank_idx, token_idx, slot_idx] = fc2.to(torch.bfloat16)

    output_fp32 = torch.zeros(
        (ranks, tokens, hidden), dtype=torch.float32, device=input_q.device
    )
    for slot in range(topk):
        valid = topk_ids[:, :, slot] != -1
        weighted = route_terms[:, :, slot].float() * topk_weights[
            :, :, slot
        ].float().unsqueeze(-1)
        output_fp32 = output_fp32 + torch.where(
            valid.unsqueeze(-1), weighted, torch.zeros_like(weighted)
        )

    return MegaMoEReferenceResult(
        route_terms=route_terms,
        output=output_fp32.to(torch.bfloat16),
    )


__all__ = [
    "BLOCK_SIZE",
    "MegaMoEReferenceResult",
    "compute_reference",
    "decode_e8m0",
    "dequantize_mxfp4",
    "dequantize_mxfp8",
    "interleave_gate_up_8",
    "quantize_mxfp4",
    "quantize_mxfp8",
]
