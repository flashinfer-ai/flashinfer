# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Torch-side helpers for the SM107 GLU mega kernel (quant, SF swizzle, reference).

The ``next/`` drop's own quant/swizzle helpers live in its *test harness*
(``tester/``), which is not vendored, so the torch-side equivalents live here
in the shim.  Semantics mirror the harness:

- MXFP8 block-32 quantization uses a round-toward-+inf (``cvt.rp.satfinite``)
  E8M0 block scale, so quantized data never overflows fp8.  The harness routes
  the rounding + reciprocal through instruction-faithful device ops; the torch
  emulation here differs by at most 1 ulp of the fp32 reciprocal, which is why
  the shim reference is compared with tolerance bands, never bitwise.
- The weight scale-factor plane is the 32x4x4 atom swizzle (``to_blocked``),
  flattened per expert — identical layout math to the harness/SM100 helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

Mxfp8BlockSize = 32
GateUpInterleave = 32  # fixed by Sm107Mxfp8GluFc12Kernel (gate/up pair stripe)
SfAtomRows = 128  # to_blocked pads rows to 128 (32x4 atoms x 4-row group)
SfAtomCols = 4

_FP8_MAX = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def _ceil_log2_exponent(x: torch.Tensor) -> torch.Tensor:
    """``ceil(log2(x))`` as int32, via exact fp32 bit decomposition.

    Pure integer/bit ops only: transcendentals like ``torch.exp2``/``log2``
    (and potentially ``frexp``) route through torch's jiterator (NVRTC), whose
    bundled compiler rejects ``sm_107``. For a positive normal fp32,
    ``ceil(log2(x)) = unbiased_exponent + (mantissa_bits != 0)``. Zero /
    non-finite inputs map to the smallest usable exponent (the data block is
    all-zero in that case anyway).
    """
    finite = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    bits = finite.clamp_min(2.0**-126).view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    return (exponent + has_mantissa.to(torch.int32)).clamp(-126, 127)


def _pow2_f32(exponent: torch.Tensor) -> torch.Tensor:
    """``2.0 ** exponent`` for int32 exponents in [-126, 127], via fp32 bits."""
    return ((exponent + 127) << 23).view(torch.float32)


def quantize_mxfp8_block32(
    tensor: torch.Tensor, data_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-quantize along the trailing dim with per-32 E8M0 block scales.

    Returns ``(data, scale)`` where ``data`` has ``tensor``'s shape in
    ``data_dtype`` and ``scale`` is E8M0 (viewed as ``float8_e8m0fnu``) with
    the trailing dim divided by 32. The scale rounds toward +inf
    (``cvt.rp.satfinite`` semantics), so data never overflows fp8.
    """
    if tensor.shape[-1] % Mxfp8BlockSize != 0:
        raise ValueError(
            f"trailing dim ({tensor.shape[-1]}) must be a multiple of {Mxfp8BlockSize}."
        )
    fp32 = tensor.to(torch.float32)
    blocked = fp32.reshape(*fp32.shape[:-1], -1, Mxfp8BlockSize)
    absmax = blocked.abs().amax(dim=-1)
    exponent = _ceil_log2_exponent(absmax * (1.0 / _FP8_MAX[data_dtype]))
    scale = _pow2_f32(exponent)
    data = (blocked / scale.unsqueeze(-1)).reshape(fp32.shape).to(data_dtype)
    e8m0 = (exponent + 127).clamp(1, 254)
    return data, e8m0.to(torch.uint8).view(torch.float8_e8m0fnu)


def e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 (or uint8-viewed) scale plane -> fp32 powers of two (bit-exact)."""
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


def to_blocked(scale_2d: torch.Tensor) -> torch.Tensor:
    """Pad and apply the 32x4x4 SF atom swizzle to one raw 2D scale (-> flat)."""
    if scale_2d.dim() != 2:
        raise ValueError(f"expected 2D scale tensor, got {scale_2d.dim()}D.")
    rows, cols = scale_2d.shape
    if rows == 0 or cols == 0:
        return scale_2d.new_empty((0,))
    row_blocks = ceil_div(rows, SfAtomRows)
    col_blocks = ceil_div(cols, SfAtomCols)
    padded_rows = row_blocks * SfAtomRows
    padded_cols = col_blocks * SfAtomCols
    padded = scale_2d
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), dtype=scale_2d.dtype, device=scale_2d.device
        )
        padded[:rows, :cols] = scale_2d
    blocks = padded.view(row_blocks, SfAtomRows, col_blocks, SfAtomCols).permute(
        0, 2, 1, 3
    )
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def swizzled_flat_sf_size(rows: int, cols: int) -> int:
    """numel of ``to_blocked`` output for a raw ``(rows, cols)`` SF plane."""
    return round_up(rows, SfAtomRows) * round_up(cols, SfAtomCols)


def interleave_gate_up_32(w13: torch.Tensor, *, intermediate_size: int) -> torch.Tensor:
    """Reorder canonical gate‖up halves into the kernel's 32-row pair stripes.

    ``w13`` is ``(..., 2*intermediate, hidden)`` with the gate rows first;
    the kernel's FC1 N axis expects ``(pair, {gate, up}, 32)`` stripes.
    """
    fc1_out = 2 * intermediate_size
    if w13.shape[-2] != fc1_out:
        raise ValueError(f"w13 N axis {w13.shape[-2]} != 2*intermediate {fc1_out}.")
    if intermediate_size % GateUpInterleave != 0:
        raise ValueError(
            f"intermediate_size ({intermediate_size}) must be a multiple of "
            f"the gate/up interleave {GateUpInterleave}."
        )
    pairs = intermediate_size // GateUpInterleave
    gate, up = w13.unflatten(-2, (2, intermediate_size)).unbind(dim=-3)
    gate = gate.unflatten(-2, (pairs, GateUpInterleave))
    up = up.unflatten(-2, (pairs, GateUpInterleave))
    return torch.stack((gate, up), dim=-3).flatten(-4, -2)


def compute_megamoe_reference_sm107_glu(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf_raw: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf_raw: torch.Tensor,
    *,
    local_expert_offset: int,
    gate_up_clamp: Optional[float],
    apply_topk_in_fc1: bool,
    num_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Pure-torch single-rank oracle for the SM107 mxfp8 GLU fprop kernel.

    Consumes the SAME staged fp8 payloads + RAW (unswizzled) scale planes the
    kernel sees, so the only divergence from the device kernel is fp32 GEMM
    accumulation order and the in-kernel FC2-input requantization (1-ulp scale
    reciprocal differences) — compare with tolerance bands.

    Weight tensors are in kernel layout: ``fc1_weight (E, hidden, 2I)`` with
    hidden stride-1 and 32-interleaved gate/up N axis, ``fc1_weight_sf_raw
    (E, 2I, hidden/32)``; ``fc2_weight (E, intermediate, hidden)`` with
    intermediate stride-1, ``fc2_weight_sf_raw (E, hidden, intermediate/32)``.
    """
    device = x_fp8.device
    tokens = x_fp8.shape[0] if num_tokens is None else num_tokens
    hidden = x_fp8.shape[1]
    num_local_experts, _, fc1_out = fc1_weight.shape
    intermediate = fc1_out // 2

    x = x_fp8[:tokens].to(torch.float32) * e8m0_to_f32(
        x_sf[:tokens, : hidden // Mxfp8BlockSize]
    ).repeat_interleave(Mxfp8BlockSize, dim=1)

    idx = topk_idx[:tokens].to(torch.int64)
    output = torch.zeros((x_fp8.shape[0], hidden), dtype=torch.float32, device=device)

    for local_e in range(num_local_experts):
        routed = (idx == local_expert_offset + local_e).nonzero(as_tuple=False)
        if routed.shape[0] == 0:
            continue
        src_t, src_k = routed[:, 0], routed[:, 1]
        w1 = fc1_weight[local_e].to(torch.float32) * e8m0_to_f32(
            fc1_weight_sf_raw[local_e]
        ).repeat_interleave(Mxfp8BlockSize, dim=1).transpose(0, 1)
        fc1 = x[src_t] @ w1  # (v, 2I): 32-interleaved (pair, {gate,up}, 32)
        pairs = fc1.view(-1, intermediate // GateUpInterleave, 2, GateUpInterleave)
        gate, up = pairs[:, :, 0, :], pairs[:, :, 1, :]
        if gate_up_clamp is not None:
            gate = gate.clamp(max=gate_up_clamp)
            up = up.clamp(min=-gate_up_clamp, max=gate_up_clamp)
        act = (up * (gate * torch.sigmoid(gate))).reshape(-1, intermediate)
        if apply_topk_in_fc1:
            act = act * topk_weights[src_t, src_k].to(torch.float32).unsqueeze(-1)
        # In-kernel FC2-input requantization (mxfp8 wire between FC1 and FC2).
        act_q, act_sf = quantize_mxfp8_block32(act, x_fp8.dtype)
        act = act_q.to(torch.float32) * e8m0_to_f32(act_sf).repeat_interleave(
            Mxfp8BlockSize, dim=1
        )
        w2 = fc2_weight[local_e].to(torch.float32) * e8m0_to_f32(
            fc2_weight_sf_raw[local_e]
        ).repeat_interleave(Mxfp8BlockSize, dim=1).transpose(0, 1)
        term = (act @ w2).to(torch.bfloat16).to(torch.float32)
        if not apply_topk_in_fc1:
            term = term * topk_weights[src_t, src_k].to(torch.float32).unsqueeze(-1)
        output.index_add_(0, src_t, term)

    return output.to(torch.bfloat16)


__all__ = [
    "GateUpInterleave",
    "Mxfp8BlockSize",
    "ceil_div",
    "compute_megamoe_reference_sm107_glu",
    "e8m0_to_f32",
    "interleave_gate_up_32",
    "quantize_mxfp8_block32",
    "round_up",
    "swizzled_flat_sf_size",
    "to_blocked",
]
