# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Torch-side helpers for the SM107 block-scaled mega kernel (quant, SF swizzle,
reference).

The ``next/`` drop's own quant/swizzle helpers live in its *test harness*
(``tester/``), which is not vendored, so the torch-side equivalents live here
in the shim.  Semantics mirror the harness:

- MXFP8 block-32 quantization uses a round-toward-+inf (``cvt.rp.satfinite``)
  E8M0 block scale, so quantized data never overflows fp8.  The harness routes
  the rounding + reciprocal through instruction-faithful device ops; the torch
  emulation here differs by at most 1 ulp of the fp32 reciprocal, which is why
  the shim reference is compared with tolerance bands, never bitwise.
- NVFP4 block-16 quantization uses an FP8-E4M3 block scale of
  ``absmax / 6 * norm_const`` and rescales by ``norm_const / round_trip(scale)``
  (the kernel epilogue's ``nvfp4_quant_impl``); the torch emulation uses exact
  fp32 division instead of ``rcp.approx.ftz`` and a distance-min E2M1 encoder
  instead of ``cvt.rn.satfinite.e2m1x2`` — again a tolerance-band, not bitwise,
  match.  All transcendental-free (torch's jiterator/NVRTC rejects sm_107).
- The weight scale-factor plane is the 32x4x4 atom swizzle (``to_blocked``),
  flattened per expert — identical layout math to the harness/SM100 helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

Mxfp8BlockSize = 32
Nvfp4BlockSize = 16
GateUpInterleave = 16  # fixed by the swap-AB gated-act epilogue (gate/up stripe)
SfAtomRows = 128  # to_blocked pads rows to 128 (32x4 atoms x 4-row group)
SfAtomCols = 4

_FP8_MAX = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}
_E2M1_MAX = 6.0
# All positive E2M1 code values (sign handled separately).
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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


def pack_f32_to_fp4(fp32: torch.Tensor) -> torch.Tensor:
    """Round fp32 to FP4 E2M1 (nearest, satfinite) and nibble-pack.

    Even trailing element -> low nibble, odd -> high (the layout
    ``cvt.rn.satfinite.e2m1x2.f32`` produces and :func:`unpack_fp4_to_f32`
    decodes). Distance-min against the 8-value magnitude table; exact
    midpoints resolve to the EVEN code, matching the instruction's
    round-to-nearest-even (e.g. 0.75 -> 1.0, 1.25 -> 1.0, 2.5 -> 2.0).
    """
    if fp32.shape[-1] % 2 != 0:
        raise ValueError(f"FP4 pack needs an even trailing dim, got {fp32.shape[-1]}.")
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=fp32.device)
    magnitude = torch.nan_to_num(fp32, nan=0.0).abs().clamp(max=_E2M1_MAX)
    distance = (magnitude.unsqueeze(-1) - values).abs()
    # Ties-to-even: among the (at most two, adjacent) codes at the minimum
    # distance, rank even codes first; non-minimal codes rank last.
    parity = torch.arange(len(_E2M1_VALUES), device=fp32.device) % 2
    tie_rank = torch.where(
        distance == distance.amin(dim=-1, keepdim=True),
        parity.to(distance.dtype).expand_as(distance),
        torch.full_like(distance, len(_E2M1_VALUES)),
    )
    code = tie_rank.argmin(dim=-1).to(torch.uint8)
    sign = (torch.signbit(fp32) & (code != 0)).to(torch.uint8) << 3
    nibble = code | sign
    pairs = nibble.reshape(*nibble.shape[:-1], -1, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return packed if fp4_dtype is None else packed.view(fp4_dtype)


def unpack_fp4_to_f32(packed: torch.Tensor) -> torch.Tensor:
    """Nibble-packed E2M1 (uint8 / float4_e2m1fn_x2) -> fp32, trailing dim doubled."""
    raw = packed.view(torch.uint8)
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=raw.device)
    low = raw & 0x0F
    high = raw >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(*raw.shape[:-1], -1)
    magnitude = values[(nibbles & 0x7).to(torch.int64)]
    return torch.where(nibbles & 0x8 != 0, -magnitude, magnitude)


def quantize_nvfp4_block16(
    tensor: torch.Tensor, norm_const: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4-quantize along the trailing dim with per-16 FP8-E4M3 block scales.

    Returns ``(packed_data, scale)``: packed E2M1 with the trailing dim halved
    and an E4M3 scale plane with the trailing dim divided by 16. Mirrors the
    kernel epilogue's ``nvfp4_quant_impl`` / the harness ``quant_nvfp4``:
    ``scale = e4m3(absmax / 6 * norm_const)``, data scaled by
    ``norm_const / round_trip(scale)`` (zero where the scale is zero).
    """
    if tensor.shape[-1] % Nvfp4BlockSize != 0:
        raise ValueError(
            f"trailing dim ({tensor.shape[-1]}) must be a multiple of {Nvfp4BlockSize}."
        )
    fp32 = tensor.to(torch.float32)
    blocked = fp32.reshape(*fp32.shape[:-1], -1, Nvfp4BlockSize)
    absmax = blocked.abs().amax(dim=-1)
    scale_f32 = (absmax * (norm_const / _E2M1_MAX)).clamp(
        min=-_FP8_MAX[torch.float8_e4m3fn], max=_FP8_MAX[torch.float8_e4m3fn]
    )
    scale_fp8 = scale_f32.to(torch.float8_e4m3fn)
    scale_rt = scale_fp8.to(torch.float32)
    acc_scale = torch.where(
        scale_rt > 0, float(norm_const) / scale_rt.clamp_min(2.0**-126), 0.0
    )
    scaled = (blocked * acc_scale.unsqueeze(-1)).reshape(fp32.shape)
    return pack_f32_to_fp4(scaled), scale_fp8


def e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 (or uint8-viewed) scale plane -> fp32 powers of two (bit-exact)."""
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


def scale_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """Any SF plane (E8M0 or E4M3, possibly uint8-viewed) -> fp32."""
    if scale.dtype == torch.float8_e4m3fn:
        return scale.to(torch.float32)
    return e8m0_to_f32(scale)


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


def interleave_gate_up_16(w13: torch.Tensor, *, intermediate_size: int) -> torch.Tensor:
    """Reorder canonical gate‖up halves into the kernel's 16-row pair stripes.

    ``w13`` is ``(..., 2*intermediate, hidden)`` with the gate rows first;
    the swap-AB gated-act epilogue's FC1 N axis expects
    ``(pair, {gate, up}, 16)`` stripes.
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


def _dequant_activation(
    quant_kind: str,
    x_data: torch.Tensor,
    x_sf: torch.Tensor,
    num_tokens: int,
    hidden: int,
) -> torch.Tensor:
    """Staged activation payload + raw per-token SF plane -> fp32 (tokens, hidden)."""
    if quant_kind == "nvfp4":
        data = unpack_fp4_to_f32(x_data[:num_tokens])[:, :hidden]
        vec = Nvfp4BlockSize
    else:
        data = x_data[:num_tokens, :hidden].to(torch.float32)
        vec = Mxfp8BlockSize
    sf = scale_to_f32(x_sf[:num_tokens, : hidden // vec])
    return data * sf.repeat_interleave(vec, dim=1)


def _dequant_weight_k_major(
    quant_kind: str, weight: torch.Tensor, sf_raw: torch.Tensor
) -> torch.Tensor:
    """K-major weight (N, K[packed]) + raw SF (N, K/vec) -> fp32 (N, K)."""
    if quant_kind == "nvfp4":
        data = unpack_fp4_to_f32(weight)
        vec = Nvfp4BlockSize
    else:
        data = weight.to(torch.float32)
        vec = Mxfp8BlockSize
    return data * scale_to_f32(sf_raw).repeat_interleave(vec, dim=1)


def _quantize_fc2_wire(
    quant_kind: str, act: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Emulate the in-kernel FC1->FC2 wire quantization for one expert."""
    if quant_kind == "nvfp4":
        return quantize_nvfp4_block16(act)
    data_dtype = (
        torch.float8_e5m2 if quant_kind == "mxfp8_e5m2" else torch.float8_e4m3fn
    )
    return quantize_mxfp8_block32(act, data_dtype)


def compute_megamoe_reference_sm107_block_scaled(
    x_data: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_weight_k_major: torch.Tensor,
    fc1_weight_sf_raw: torch.Tensor,
    fc2_weight_k_major: torch.Tensor,
    fc2_weight_sf_raw: torch.Tensor,
    *,
    quant_kind: str,
    local_expert_offset: int,
    gate_up_clamp: Optional[float],
    apply_topk_at_fc1: bool,
    num_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Pure-torch single-rank oracle for the SM107 block-scaled inference kernel.

    Consumes the SAME staged quantized payloads + RAW (unswizzled) scale planes
    the kernel sees, so the only divergence from the device kernel is fp32 GEMM
    accumulation order and the in-kernel FC2-input requantization (approximate
    reciprocal / E2M1 tie rounding) — compare with tolerance bands.

    Weight tensors are K-MAJOR PHYSICAL layout (what the transforms build
    before the logical transpose view): ``fc1_weight_k_major
    (E, 2I, hidden[packed])`` with the 16-interleaved gate/up N axis and raw SF
    ``(E, 2I, hidden/vec)``; ``fc2_weight_k_major (E, hidden, I[packed])`` with
    raw SF ``(E, hidden, I/vec)``.
    """
    device = topk_idx.device
    tokens = x_data.shape[0] if num_tokens is None else num_tokens
    num_local_experts, fc1_out, _ = fc1_weight_k_major.shape
    hidden = fc2_weight_k_major.shape[1]
    intermediate = fc1_out // 2

    x = _dequant_activation(quant_kind, x_data, x_sf, tokens, hidden)

    idx = topk_idx[:tokens].to(torch.int64)
    output = torch.zeros((x_data.shape[0], hidden), dtype=torch.float32, device=device)

    for local_e in range(num_local_experts):
        routed = (idx == local_expert_offset + local_e).nonzero(as_tuple=False)
        if routed.shape[0] == 0:
            continue
        src_t, src_k = routed[:, 0], routed[:, 1]
        w1 = _dequant_weight_k_major(
            quant_kind, fc1_weight_k_major[local_e], fc1_weight_sf_raw[local_e]
        )
        fc1 = x[src_t] @ w1.transpose(0, 1)  # (v, 2I): (pair, {gate,up}, 16)
        pairs = fc1.view(-1, intermediate // GateUpInterleave, 2, GateUpInterleave)
        gate, up = pairs[:, :, 0, :], pairs[:, :, 1, :]
        if gate_up_clamp is not None:
            gate = gate.clamp(max=gate_up_clamp)
            up = up.clamp(min=-gate_up_clamp, max=gate_up_clamp)
        act = (up * (gate * torch.sigmoid(gate))).reshape(-1, intermediate)
        if apply_topk_at_fc1:
            act = act * topk_weights[src_t, src_k].to(torch.float32).unsqueeze(-1)
        # In-kernel FC2-input requantization (the FC1->FC2 wire format).
        act_q, act_sf = _quantize_fc2_wire(quant_kind, act)
        vec = Nvfp4BlockSize if quant_kind == "nvfp4" else Mxfp8BlockSize
        if quant_kind == "nvfp4":
            act = unpack_fp4_to_f32(act_q)
        else:
            act = act_q.to(torch.float32)
        act = act * scale_to_f32(act_sf).repeat_interleave(vec, dim=1)
        w2 = _dequant_weight_k_major(
            quant_kind, fc2_weight_k_major[local_e], fc2_weight_sf_raw[local_e]
        )
        term = (act @ w2.transpose(0, 1)).to(torch.bfloat16).to(torch.float32)
        if not apply_topk_at_fc1:
            term = term * topk_weights[src_t, src_k].to(torch.float32).unsqueeze(-1)
        output.index_add_(0, src_t, term)

    return output.to(torch.bfloat16)


__all__ = [
    "GateUpInterleave",
    "Mxfp8BlockSize",
    "Nvfp4BlockSize",
    "ceil_div",
    "compute_megamoe_reference_sm107_block_scaled",
    "e8m0_to_f32",
    "interleave_gate_up_16",
    "pack_f32_to_fp4",
    "quantize_mxfp8_block32",
    "quantize_nvfp4_block16",
    "round_up",
    "scale_to_f32",
    "swizzled_flat_sf_size",
    "to_blocked",
    "unpack_fp4_to_f32",
]
