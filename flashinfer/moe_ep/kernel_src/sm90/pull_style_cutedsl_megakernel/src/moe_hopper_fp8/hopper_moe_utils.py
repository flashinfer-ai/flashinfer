# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-side utility helpers for the FP8 Hopper MoE path."""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch

from common.megamoe_constants import (
    Fp8BlockScaleK,
    Fp8Fc2ActivationScaleK,
    Fp8WeightScaleBlockK,
    Fp8WeightScaleBlockN,
)

FP8_KIND_CHOICES = ("fp8_e4m3", "fp8_e5m2", "mxfp8_e4m3", "mxfp8_e5m2")
FP8_SCALE_MODE_CHOICES = ("per_tensor", "blockwise")
FP8_ACCUM_MODE_CHOICES = ("1xacc", "2xacc")
Fp8PerTensorTargetAmax = 0.25
Fp8PerTensorScaleEpsilon = 1.0e-12
Fp8PerTensorOutputQuantMargin = 0.95
Fp8BlockScaleEpsilon = 1.0e-30


def fp8_kind_to_cutlass_dtype(kind: str):
    """Map supported Hopper FP8 kind strings to Cutlass dtypes."""
    import cutlass

    return {
        "fp8_e4m3": cutlass.Float8E4M3FN,
        "mxfp8_e4m3": cutlass.Float8E4M3FN,
        "fp8_e5m2": cutlass.Float8E5M2,
        "mxfp8_e5m2": cutlass.Float8E5M2,
    }[kind]


def make_fp8_per_tensor_dequant_scale(
    source: Union[torch.Tensor, torch.dtype],
    shape: Optional[Tuple[int, ...]] = None,
    *,
    reduce_dims: Optional[Union[int, Tuple[int, ...]]] = None,
    target_amax: float = Fp8PerTensorTargetAmax,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Return FP32 dequant scale for synthetic per-tensor FP8 test payloads.

    If ``source`` is a tensor, derive scale from that tensor's observed absmax.
    If ``source`` is a dtype, return the perf-only constant scale for ``shape``.
    """
    if isinstance(source, torch.Tensor):
        tensor = source
        if tensor.numel() == 0:
            return torch.ones((1,), dtype=torch.float32, device=tensor.device)

        abs_values = tensor.to(torch.float32).abs()
        if reduce_dims is None:
            absmax = abs_values.amax().reshape(1)
        else:
            absmax = abs_values.amax(dim=reduce_dims)

        scale = float(target_amax) / torch.clamp(
            absmax, min=Fp8PerTensorScaleEpsilon
        )
        return scale.to(dtype=torch.float32, device=tensor.device)

    if not isinstance(source, torch.dtype):
        raise TypeError(
            f"source must be a torch.Tensor or torch.dtype, got {source!r}."
        )
    if shape is None:
        raise ValueError("shape must be provided when source is a torch.dtype.")
    if device is None:
        device = "cuda"
    scale_val = float(target_amax) / fp8_dtype_max(source)
    return torch.full(shape, scale_val, dtype=torch.float32, device=device)


def compute_fp8_per_tensor_output_dequant_scale_from_absmax(
    absmax,
    fp8_dtype: torch.dtype,
    *,
    device: Optional[torch.device] = None,
    margin: float = Fp8PerTensorOutputQuantMargin,
) -> torch.Tensor:
    """Return scalar fp32 scale used to quantize an fp32 activation to fp8."""
    if isinstance(absmax, torch.Tensor):
        if device is None:
            device = absmax.device
        absmax_val = float(absmax.detach().to(torch.float32).amax().item())
    else:
        absmax_val = float(absmax)

    fp8_amax = float(torch.finfo(fp8_dtype).max) * float(margin)
    if absmax_val <= 0.0:
        scale_val = 1.0
    else:
        scale_val = max(absmax_val / fp8_amax, Fp8PerTensorScaleEpsilon)

    return torch.tensor((scale_val,), dtype=torch.float32, device=device)


def _check_2d_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be a 2D tensor, got {tensor.dim()}D.")


def _check_divisible(name: str, value: int, divisor: int) -> None:
    if divisor <= 0:
        raise ValueError(f"{name} divisor must be positive, got {divisor}.")
    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by {divisor}.")


def fp8_dtype_max(fp8_dtype: torch.dtype) -> float:
    """Return the finite max value for a supported torch FP8 dtype."""
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"Unsupported fp8 dtype: {fp8_dtype}.")
    return float(torch.finfo(fp8_dtype).max)


def make_constant_block_scale(
    fp8_dtype: torch.dtype,
    shape: Tuple[int, ...],
    *,
    target_amax: float = Fp8PerTensorTargetAmax,
    device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """Small positive FP32 block scale for perf-only synthetic FP8 bytes."""
    scale_val = float(target_amax) / fp8_dtype_max(fp8_dtype)
    return torch.full(shape, scale_val, dtype=torch.float32, device=device)


def create_fp8_tensor(
    shape: Tuple[int, ...],
    fp8_dtype: torch.dtype,
    *,
    perf_run: bool,
    nonzero_prob: float = 0.20,
    nonzero_value: float = 1.0,
    positive_prob: Optional[float] = None,
    negative_prob: Optional[float] = None,
    return_fp8: bool = True,
    device: Union[str, torch.device] = "cuda",
    generator: Optional[torch.Generator] = None,
    perf_positive_only: bool = False,
) -> torch.Tensor:
    """Create sparse source data and optionally return it as FP8 payload.

    Correctness mode uses sparse signed FP32 source data.  By default,
    ``nonzero_prob`` is split evenly across ``+nonzero_value`` and
    ``-nonzero_value``.  Perf mode can generate random finite FP8 bytes directly
    to avoid large temporary FP32 tensors; ``return_fp8=False`` always returns
    the FP32 source.
    """
    if not 0.0 <= float(nonzero_prob) <= 1.0:
        raise ValueError(f"nonzero_prob must be in [0, 1], got {nonzero_prob}.")
    if positive_prob is None and negative_prob is None:
        pos_prob = float(nonzero_prob) * 0.5
        neg_prob = float(nonzero_prob) * 0.5
    elif positive_prob is not None and negative_prob is not None:
        pos_prob = float(positive_prob)
        neg_prob = float(negative_prob)
    else:
        raise ValueError("positive_prob and negative_prob must be provided together.")
    if pos_prob < 0.0 or neg_prob < 0.0:
        raise ValueError(
            f"positive_prob and negative_prob must be non-negative, got "
            f"{pos_prob} and {neg_prob}."
        )
    if perf_run and return_fp8:
        n = 1
        for s in shape:
            n *= s
        if perf_positive_only:
            flat_bytes = torch.empty((n,), dtype=torch.uint8, device=device)
            if fp8_dtype == torch.float8_e4m3fn:
                flat_bytes.random_(0, 127, generator=generator)
            elif fp8_dtype == torch.float8_e5m2:
                flat_bytes.random_(0, 124, generator=generator)
            else:
                raise ValueError(f"Unsupported fp8 dtype: {fp8_dtype}")
        elif fp8_dtype == torch.float8_e4m3fn:
            idx = torch.randint(0, 254, (n,), device=device, generator=generator)
            flat_bytes = torch.where(idx < 127, idx, idx + 1).to(torch.uint8)
        elif fp8_dtype == torch.float8_e5m2:
            idx = torch.randint(0, 248, (n,), device=device, generator=generator)
            flat_bytes = torch.where(idx < 124, idx, idx + 4).to(torch.uint8)
        else:
            raise ValueError(f"Unsupported fp8 dtype: {fp8_dtype}")
        return flat_bytes.view(fp8_dtype).reshape(shape)

    pos_threshold = pos_prob
    neg_threshold = pos_prob + neg_prob
    fp32 = torch.zeros(shape, dtype=torch.float32, device=device)
    rand = torch.rand(shape, device=device, generator=generator)
    fp32[rand < pos_threshold] = float(nonzero_value)
    fp32[(rand >= pos_threshold) & (rand < neg_threshold)] = -float(nonzero_value)
    if return_fp8:
        return fp32.to(fp8_dtype)
    return fp32


def quantize_fp8_per_token_block(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    block_k: int = Fp8BlockScaleK,
    scale_epsilon: float = Fp8BlockScaleEpsilon,
    target_amax: Optional[float] = None,
    use_reciprocal_multiply: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D activation with one fp32 scale per row and K block.

    By default the scale is ``absmax / fp8_max`` for output requantization.
    For synthetic blockwise input payloads, pass ``target_amax`` to keep the
    input values as the FP8 payload and use the per-tensor-style dequant scale
    ``target_amax / observed_payload_absmax``.
    """
    _check_2d_tensor("tensor", tensor)
    rows, cols = tensor.shape
    _check_divisible("tensor.shape[1]", cols, block_k)

    fp32 = tensor.to(torch.float32)
    blocks = fp32.reshape(rows, cols // block_k, block_k)
    if target_amax is None:
        absmax = blocks.abs().amax(dim=-1)
        scale = absmax / fp8_dtype_max(fp8_dtype)
        scale = torch.clamp(scale, min=float(scale_epsilon)).to(torch.float32)
        scale_view = scale.unsqueeze(-1)
        scaled_blocks = (
            blocks * torch.reciprocal(scale_view)
            if use_reciprocal_multiply
            else blocks / scale_view
        )
        scaled = scaled_blocks.reshape(rows, cols)
        quant = scaled.to(fp8_dtype)
    else:
        quant_blocks = blocks.to(fp8_dtype)
        absmax = quant_blocks.to(torch.float32).abs().amax(dim=-1)
        scale = float(target_amax) / torch.clamp(
            absmax, min=Fp8PerTensorScaleEpsilon
        )
        scale = scale.to(torch.float32)
        quant = quant_blocks.reshape(rows, cols)
    return quant, scale


def quantize_fp8_with_per_token_block_scale(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    block_k: int = Fp8BlockScaleK,
    use_reciprocal_multiply: bool = False,
) -> torch.Tensor:
    """Quantize a 2D activation using an existing per-row, per-K-block scale."""
    _check_2d_tensor("tensor", tensor)
    _check_2d_tensor("scale", scale)
    rows, cols = tensor.shape
    _check_divisible("tensor.shape[1]", cols, block_k)
    expected = (rows, cols // block_k)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape mismatch: expected {expected}, got {tuple(scale.shape)}."
        )
    scale_expanded = scale.to(torch.float32).repeat_interleave(block_k, dim=1)
    tensor_fp32 = tensor.to(torch.float32)
    scaled = (
        tensor_fp32 * torch.reciprocal(scale_expanded)
        if use_reciprocal_multiply
        else tensor_fp32 / scale_expanded
    )
    return scaled.to(fp8_dtype)


def dequantize_fp8_per_token_block(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_k: int = Fp8BlockScaleK,
) -> torch.Tensor:
    """Dequantize a 2D activation with one scale per row and K block."""
    _check_2d_tensor("tensor", tensor)
    _check_2d_tensor("scale", scale)
    rows, cols = tensor.shape
    _check_divisible("tensor.shape[1]", cols, block_k)
    expected = (rows, cols // block_k)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape mismatch: expected {expected}, got {tuple(scale.shape)}."
        )
    scale_expanded = scale.to(torch.float32).repeat_interleave(block_k, dim=1)
    return tensor.to(torch.float32) * scale_expanded


def quantize_fp8_weight_block_nk(
    weight_nk: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    block_n: int = Fp8WeightScaleBlockN,
    block_k: int = Fp8WeightScaleBlockK,
    scale_epsilon: float = Fp8BlockScaleEpsilon,
    target_amax: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a logical ``(N, K)`` weight with 128x128 fp32 block scales."""
    _check_2d_tensor("weight_nk", weight_nk)
    n, k = weight_nk.shape
    _check_divisible("weight_nk.shape[0]", n, block_n)
    _check_divisible("weight_nk.shape[1]", k, block_k)

    fp32 = weight_nk.to(torch.float32)
    blocks = fp32.reshape(n // block_n, block_n, k // block_k, block_k)
    if target_amax is None:
        absmax = blocks.abs().amax(dim=(1, 3))
        scale = absmax / fp8_dtype_max(fp8_dtype)
        scale = torch.clamp(scale, min=float(scale_epsilon)).to(torch.float32)
        quant = (blocks / scale[:, None, :, None]).reshape(n, k).to(fp8_dtype)
    else:
        quant_blocks = blocks.to(fp8_dtype)
        absmax = quant_blocks.to(torch.float32).abs().amax(dim=(1, 3))
        scale = float(target_amax) / torch.clamp(
            absmax, min=Fp8PerTensorScaleEpsilon
        )
        scale = scale.to(torch.float32)
        quant = quant_blocks.reshape(n, k)
    return quant, scale


def dequantize_fp8_weight_block(
    weight: torch.Tensor,
    scale_nk: torch.Tensor,
    *,
    weight_layout: Literal["kn", "nk"] = "kn",
    block_n: int = Fp8WeightScaleBlockN,
    block_k: int = Fp8WeightScaleBlockK,
) -> torch.Tensor:
    """Dequantize FP8 weight using logical ``(N-block, K-block)`` scales."""
    _check_2d_tensor("weight", weight)
    _check_2d_tensor("scale_nk", scale_nk)

    if weight_layout == "kn":
        k, n = weight.shape
        expected = (n // block_n, k // block_k)
        _check_divisible("weight.shape[0]", k, block_k)
        _check_divisible("weight.shape[1]", n, block_n)
        if tuple(scale_nk.shape) != expected:
            raise ValueError(
                f"scale_nk shape mismatch: expected {expected}, "
                f"got {tuple(scale_nk.shape)}."
            )
        scale = (
            scale_nk.to(torch.float32)
            .transpose(0, 1)
            .repeat_interleave(block_k, dim=0)
            .repeat_interleave(block_n, dim=1)
        )
    elif weight_layout == "nk":
        n, k = weight.shape
        expected = (n // block_n, k // block_k)
        _check_divisible("weight.shape[0]", n, block_n)
        _check_divisible("weight.shape[1]", k, block_k)
        if tuple(scale_nk.shape) != expected:
            raise ValueError(
                f"scale_nk shape mismatch: expected {expected}, "
                f"got {tuple(scale_nk.shape)}."
            )
        scale = (
            scale_nk.to(torch.float32)
            .repeat_interleave(block_n, dim=0)
            .repeat_interleave(block_k, dim=1)
        )
    else:
        raise ValueError(f"Unsupported weight_layout: {weight_layout!r}.")

    return weight.to(torch.float32) * scale


def fp8_block_scaled_reference_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    a_scale_block_k: int = Fp8BlockScaleK,
    b_scale_block_n: int = Fp8WeightScaleBlockN,
    b_scale_block_k: int = Fp8WeightScaleBlockK,
) -> torch.Tensor:
    """Reference block-scale FP8 matmul with promotion at scale-block boundaries."""
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"a and b must be 2D, got {a.dim()}D and {b.dim()}D.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Inner dimension mismatch: a.shape={tuple(a.shape)}, "
            f"b.shape={tuple(b.shape)}."
        )

    m, k = a.shape
    _, n = b.shape
    _check_divisible("a.shape[1]", k, a_scale_block_k)
    _check_divisible("b.shape[0]", k, b_scale_block_k)
    _check_divisible("b.shape[1]", n, b_scale_block_n)
    expected_a_scale = (m, k // a_scale_block_k)
    expected_b_scale = (n // b_scale_block_n, k // b_scale_block_k)
    if tuple(a_scale.shape) != expected_a_scale:
        raise ValueError(
            f"a_scale shape mismatch: expected {expected_a_scale}, "
            f"got {tuple(a_scale.shape)}."
        )
    if tuple(b_scale.shape) != expected_b_scale:
        raise ValueError(
            f"b_scale shape mismatch: expected {expected_b_scale}, "
            f"got {tuple(b_scale.shape)}."
        )
    if b.stride(0) != 1:
        b = b.t().contiguous().t()

    boundaries = {0, k}
    boundaries.update(range(a_scale_block_k, k, a_scale_block_k))
    boundaries.update(range(b_scale_block_k, k, b_scale_block_k))
    sorted_boundaries = sorted(boundaries)

    acc = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    one = torch.ones((), dtype=torch.float32, device=a.device)
    a_scale_fp32 = a_scale.to(device=a.device, dtype=torch.float32)
    b_scale_fp32 = b_scale.to(device=a.device, dtype=torch.float32)
    for k0, k1 in zip(sorted_boundaries[:-1], sorted_boundaries[1:]):
        a_block = k0 // a_scale_block_k
        b_block = k0 // b_scale_block_k
        raw = torch._scaled_mm(
            a[:, k0:k1].contiguous(),
            b[k0:k1, :],
            one,
            one,
            out_dtype=torch.float32,
        )
        n_scale = b_scale_fp32[:, b_block].repeat_interleave(b_scale_block_n)
        acc.add_(
            raw
            * a_scale_fp32[:, a_block].reshape(m, 1)
            * n_scale.reshape(1, n)
        )
    return acc


def _fp8_per_tensor_wgmma_reference_mm_1xacc(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Use one fast FP32 accumulator over the full K dimension."""
    one = torch.ones((), dtype=torch.float32, device=a.device)
    return torch._scaled_mm(
        a.contiguous(),
        b,
        one,
        one,
        out_dtype=torch.float32,
        use_fast_accum=True,
    )


def _fp8_per_tensor_wgmma_reference_mm_2xacc(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    k_chunk: int,
) -> torch.Tensor:
    """Promote one raw FP8 accumulator into a long-lived one per K chunk."""
    if k_chunk <= 0:
        raise ValueError(f"k_chunk must be positive, got {k_chunk}.")
    output = torch.zeros(
        (a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device
    )
    one = torch.ones((), dtype=torch.float32, device=a.device)
    for k0 in range(0, a.shape[1], k_chunk):
        k1 = min(k0 + k_chunk, a.shape[1])
        output.add_(
            torch._scaled_mm(
                a[:, k0:k1].contiguous(),
                b[k0:k1, :],
                one,
                one,
                out_dtype=torch.float32,
            )
        )
    return output


def fp8_per_tensor_wgmma_reference_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    accum_mode: str = "1xacc",
    k_chunk: int = 128,
) -> torch.Tensor:
    """Dispatch to the selected per-tensor WGMMA accumulation model."""
    if accum_mode == "1xacc":
        return _fp8_per_tensor_wgmma_reference_mm_1xacc(a, b)
    if accum_mode == "2xacc":
        return _fp8_per_tensor_wgmma_reference_mm_2xacc(a, b, k_chunk=k_chunk)
    raise ValueError(
        f"accum_mode must be one of {FP8_ACCUM_MODE_CHOICES}, got {accum_mode!r}."
    )


__all__ = [
    "FP8_KIND_CHOICES",
    "FP8_SCALE_MODE_CHOICES",
    "FP8_ACCUM_MODE_CHOICES",
    "Fp8PerTensorTargetAmax",
    "Fp8PerTensorScaleEpsilon",
    "Fp8PerTensorOutputQuantMargin",
    "Fp8BlockScaleEpsilon",
    "fp8_kind_to_cutlass_dtype",
    "compute_fp8_per_tensor_output_dequant_scale_from_absmax",
    "fp8_dtype_max",
    "make_constant_block_scale",
    "make_fp8_per_tensor_dequant_scale",
    "create_fp8_tensor",
    "quantize_fp8_per_token_block",
    "quantize_fp8_with_per_token_block_scale",
    "dequantize_fp8_per_token_block",
    "quantize_fp8_weight_block_nk",
    "dequantize_fp8_weight_block",
    "fp8_block_scaled_reference_mm",
    "fp8_per_tensor_wgmma_reference_mm",
]
