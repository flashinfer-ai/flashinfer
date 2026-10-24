# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP4/E8M0-K32 weight preprocessing for native MXFP8 MegaMoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ......weights import MoEWeightPack, PrequantizedMoEWeights

if TYPE_CHECKING:
    import torch


MXFP4_BLOCK_SIZE = 32

TransformedMegaWeights = Tuple[
    Tuple["torch.Tensor", "torch.Tensor"],
    Tuple["torch.Tensor", "torch.Tensor"],
]


def _is_packed_mxfp4(tensor: "torch.Tensor") -> bool:
    import torch

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return tensor.dtype == torch.uint8 or (
        fp4_dtype is not None and tensor.dtype == fp4_dtype
    )


def _as_kernel_fp4(tensor: "torch.Tensor") -> "torch.Tensor":
    import torch

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return (
        tensor
        if fp4_dtype is None or tensor.dtype == fp4_dtype
        else tensor.view(fp4_dtype)
    )


def _kernel_fp4_dtype() -> "torch.dtype":
    import torch

    return getattr(torch, "float4_e2m1fn_x2", torch.uint8)


def _interleave_gate_up_8(
    tensor: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    """Convert canonical gate||up rows to gate8,up8 alternating blocks."""

    if intermediate_size % MXFP4_BLOCK_SIZE:
        raise ValueError("MXFP4 MegaMoE requires intermediate_size divisible by 32")
    if tensor.ndim != 3 or tensor.shape[1] != 2 * intermediate_size:
        raise ValueError(
            f"expected FC1 shape (experts, {2 * intermediate_size}, ...), "
            f"got {tuple(tensor.shape)}"
        )
    gate = tensor[:, :intermediate_size].contiguous()
    up = tensor[:, intermediate_size:].contiguous()
    blocks = intermediate_size // 8
    out = tensor.new_empty(tensor.shape)
    out_view = out.view(tensor.shape[0], blocks, 2, 8, tensor.shape[2])
    out_view[:, :, 0].copy_(gate.view(tensor.shape[0], blocks, 8, tensor.shape[2]))
    out_view[:, :, 1].copy_(up.view(tensor.shape[0], blocks, 8, tensor.shape[2]))
    return out.contiguous()


def _quantize_expert_weights(
    weight_nk: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize one row-major expert to packed E2M1 + linear UE8M0/K32."""

    from flashinfer.quantization.fp4_quantization import mxfp4_quantize
    from flashinfer.tllm_enums import SfLayout

    return mxfp4_quantize(weight_nk, sfLayout=SfLayout.layout_linear)


def _swizzle_expert_scales(raw_sf: "torch.Tensor") -> "torch.Tensor":
    """Convert logical (N,K/32) UE8M0 bytes to the kernel's 128x4 layout."""

    from ......kernel_src.sm100.cutedsl_megamoe import to_blocked

    return to_blocked(raw_sf)


def _stack_scales(parts: list["torch.Tensor"]) -> "torch.Tensor":
    import torch

    if not parts:
        raise ValueError("at least one local expert is required")
    return torch.stack([part.view(torch.uint8) for part in parts], dim=0)


def _validate_canonical_bf16(
    weights: "MoEWeightPack",
    *,
    logical_w13_shape: tuple[int, ...],
    logical_w2_shape: tuple[int, ...],
) -> None:
    import torch

    if tuple(weights.w13.shape) != logical_w13_shape:
        raise ValueError(
            f"w13 must have shape {logical_w13_shape}, got {tuple(weights.w13.shape)}"
        )
    if tuple(weights.w2.shape) != logical_w2_shape:
        raise ValueError(
            f"w2 must have shape {logical_w2_shape}, got {tuple(weights.w2.shape)}"
        )
    if weights.w13.dtype != torch.bfloat16 or weights.w2.dtype != torch.bfloat16:
        raise TypeError(
            "canonical MXFP8 x MXFP4 MegaMoE weights must be BF16; got "
            f"{weights.w13.dtype} / {weights.w2.dtype}"
        )
    if weights.w13.device != weights.w2.device:
        raise ValueError("w13 and w2 must be on the same device")
    if not weights.w13.is_cuda:
        raise ValueError("canonical BF16 weights must be CUDA tensors")


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """Build K-major packed weights and atom-swizzled UE8M0 scale planes.

    Canonical input is BF16 ``w13[E,2I,H]`` / ``w2[E,H,I]``.  A
    :class:`PrequantizedMoEWeights` instead carries packed E2M1 data in those
    same logical layouts (with the K dimension halved) and unswizzled uint8
    UE8M0 scales.  Quantized bytes are consumed verbatim.
    """

    import torch

    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError(
            "MXFP8 x MXFP4 requires hidden and intermediate multiples of 128"
        )

    num_experts = weights.w13.shape[0]
    fc1_out = 2 * intermediate_size
    logical_w13_shape = (num_experts, fc1_out, hidden_size)
    logical_w2_shape = (num_experts, hidden_size, intermediate_size)
    packed_w13_shape = (num_experts, fc1_out, hidden_size // 2)
    packed_w2_shape = (num_experts, hidden_size, intermediate_size // 2)
    w13_sf_shape = (num_experts, fc1_out, hidden_size // MXFP4_BLOCK_SIZE)
    w2_sf_shape = (
        num_experts,
        hidden_size,
        intermediate_size // MXFP4_BLOCK_SIZE,
    )

    if isinstance(weights, PrequantizedMoEWeights):
        if (
            tuple(weights.w13.shape) != packed_w13_shape
            or tuple(weights.w2.shape) != packed_w2_shape
        ):
            raise ValueError(
                "prequantized MXFP4 weights must have packed shapes "
                f"{packed_w13_shape} / {packed_w2_shape}; got "
                f"{tuple(weights.w13.shape)} / {tuple(weights.w2.shape)}"
            )
        if not _is_packed_mxfp4(weights.w13) or not _is_packed_mxfp4(weights.w2):
            raise TypeError(
                "prequantized MXFP4 weights must be uint8 or float4_e2m1fn_x2"
            )
        if weights.w13.dtype != weights.w2.dtype:
            raise TypeError("w13 and w2 packed MXFP4 dtypes must match")
        if (
            weights.w13_scale.dtype != torch.uint8
            or weights.w2_scale.dtype != torch.uint8
        ):
            raise TypeError("MXFP4 E8M0 scale tensors must have dtype torch.uint8")
        if tuple(weights.w13_scale.shape) != w13_sf_shape:
            raise ValueError(
                f"w13_scale must have shape {w13_sf_shape}, got "
                f"{tuple(weights.w13_scale.shape)}"
            )
        if tuple(weights.w2_scale.shape) != w2_sf_shape:
            raise ValueError(
                f"w2_scale must have shape {w2_sf_shape}, got "
                f"{tuple(weights.w2_scale.shape)}"
            )
        tensors = (weights.w13, weights.w2, weights.w13_scale, weights.w2_scale)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("prequantized MXFP4 data and scales must share one device")
        w13_packed = _interleave_gate_up_8(
            weights.w13, intermediate_size=intermediate_size
        )
        w13_sf = _interleave_gate_up_8(
            weights.w13_scale, intermediate_size=intermediate_size
        )
        w2_packed = weights.w2
        w2_sf = weights.w2_scale
    else:
        _validate_canonical_bf16(
            weights,
            logical_w13_shape=logical_w13_shape,
            logical_w2_shape=logical_w2_shape,
        )
        w13_bf16 = _interleave_gate_up_8(
            weights.w13, intermediate_size=intermediate_size
        )
        w13_q: list[torch.Tensor] = []
        w13_sf_parts: list[torch.Tensor] = []
        w2_q: list[torch.Tensor] = []
        w2_sf_parts: list[torch.Tensor] = []
        for expert in range(num_experts):
            q1, sf1 = _quantize_expert_weights(w13_bf16[expert])
            q2, sf2 = _quantize_expert_weights(weights.w2[expert])
            w13_q.append(q1)
            w13_sf_parts.append(sf1)
            w2_q.append(q2)
            w2_sf_parts.append(sf2)
        w13_packed = torch.stack(w13_q, dim=0)
        w13_sf = torch.stack(w13_sf_parts, dim=0)
        w2_packed = torch.stack(w2_q, dim=0)
        w2_sf = torch.stack(w2_sf_parts, dim=0)

    fc1_weight = _as_kernel_fp4(w13_packed.transpose(1, 2))
    fc2_weight = _as_kernel_fp4(w2_packed.transpose(1, 2))
    fc1_weight_sf = _stack_scales(
        [_swizzle_expert_scales(w13_sf[e]) for e in range(num_experts)]
    )
    fc2_weight_sf = _stack_scales(
        [_swizzle_expert_scales(w2_sf[e]) for e in range(num_experts)]
    )
    return (fc1_weight, fc1_weight_sf), (fc2_weight, fc2_weight_sf)


def _swizzled_flat_sf_size(rows: int, cols: int) -> int:
    import torch

    return _swizzle_expert_scales(torch.zeros(rows, cols, dtype=torch.uint8)).numel()


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    """Validate the exact native-kernel weight ABI without touching contents."""

    import torch

    from ......core.validation.common import MoEEpConfigError
    from ...weight_validation import (
        check_transformed_mega_weights_structure,
        check_transformed_weight_pair,
    )

    if world_size <= 0 or num_experts % world_size:
        raise MoEEpConfigError("num_experts must be divisible by positive world_size")
    local_experts = num_experts // world_size
    fc1_out = 2 * intermediate_size
    fc1_sf = _swizzled_flat_sf_size(fc1_out, hidden_size // MXFP4_BLOCK_SIZE)
    fc2_sf = _swizzled_flat_sf_size(hidden_size, intermediate_size // MXFP4_BLOCK_SIZE)
    weight_dtype = _kernel_fp4_dtype()

    check_transformed_mega_weights_structure(transformed)
    check_transformed_weight_pair(
        transformed[0],
        label="fc1",
        num_local_experts=local_experts,
        weight_dtype=weight_dtype,
        expected_weight_shape=(local_experts, hidden_size // 2, fc1_out),
        scale_dtype=torch.uint8,
        expected_scale_shape=(local_experts, fc1_sf),
    )
    check_transformed_weight_pair(
        transformed[1],
        label="fc2",
        num_local_experts=local_experts,
        weight_dtype=weight_dtype,
        expected_weight_shape=(local_experts, intermediate_size // 2, hidden_size),
        scale_dtype=torch.uint8,
        expected_scale_shape=(local_experts, fc2_sf),
    )
    for label, tensor in (("fc1", transformed[0][0]), ("fc2", transformed[1][0])):
        if tensor.stride(1) != 1:
            raise MoEEpConfigError(
                f"{label} MXFP4 weight must be packed-K-major (stride(1)==1), "
                f"got strides {tensor.stride()}"
            )


__all__ = [
    "MXFP4_BLOCK_SIZE",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
