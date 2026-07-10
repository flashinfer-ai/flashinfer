"""Mega-path NVFP4 weight preprocessing for CuTeDSL MegaMoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from .....weights import MoEWeightPack

if TYPE_CHECKING:
    import torch

TransformedMegaWeights = Tuple[
    Tuple["torch.Tensor", "torch.Tensor"],
    Tuple["torch.Tensor", "torch.Tensor"],
]


def _require_cutedsl_paths() -> None:
    from ..cutedsl_backend_kernels import bootstrap_paths

    bootstrap_paths()


def _resolve_gate_up_clamp(
    *,
    gate_up_clamp: float | None,
    activation_clamp: float | None,
) -> float:
    if gate_up_clamp is not None and activation_clamp is not None:
        if gate_up_clamp != activation_clamp:
            raise ValueError(
                "gate_up_clamp and activation_clamp disagree "
                f"({gate_up_clamp} vs {activation_clamp}); pass only one."
            )
    if gate_up_clamp is not None:
        return gate_up_clamp
    if activation_clamp is not None:
        return activation_clamp
    return 1.0


def _quantize_expert_weights(
    weight_k_major: "torch.Tensor",
    *,
    norm_const: float,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Return NVFP4 weight + raw fp8 block scales with K on the trailing dim."""
    import torch

    _require_cutedsl_paths()
    from moe_nvfp4_swapab.runner_common import nvfp4_quantize_per_block_16

    return nvfp4_quantize_per_block_16(weight_k_major.to(torch.float32), norm_const)


def _swizzle_expert_scales(raw_sf: "torch.Tensor") -> "torch.Tensor":
    _require_cutedsl_paths()
    from moe_nvfp4_swapab.runner_common import to_blocked

    return to_blocked(raw_sf)


def _is_packed_nvfp4_weight(weight: "torch.Tensor") -> bool:
    import torch

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return weight.dtype == torch.uint8 or (
        fp4_dtype is not None and weight.dtype == fp4_dtype
    )


def _as_fp4_weight(weight: "torch.Tensor") -> "torch.Tensor":
    import torch

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError("torch.float4_e2m1fn_x2 is required for NVFP4 MegaMOE")
    if weight.dtype == fp4_dtype:
        return weight
    return weight.view(fp4_dtype)


def _interleave_gate_up_16(
    tensor: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    if intermediate_size % 16 != 0:
        raise ValueError(
            "NVFP4 MegaMOE requires intermediate_size to be divisible by 16, "
            f"got {intermediate_size}."
        )
    if tensor.shape[1] != 2 * intermediate_size:
        raise ValueError(
            "expected concatenated FC1 tensor with shape "
            f"(local_experts, {2 * intermediate_size}, ...), got {tuple(tensor.shape)}"
        )

    gate = tensor[:, :intermediate_size, :].contiguous()
    up = tensor[:, intermediate_size:, :].contiguous()
    num_pairs = intermediate_size // 16
    out = tensor.new_empty(tensor.shape)
    out_view = out.view(tensor.shape[0], num_pairs, 2, 16, tensor.shape[2])
    gate_view = gate.view(tensor.shape[0], num_pairs, 16, tensor.shape[2])
    up_view = up.view(tensor.shape[0], num_pairs, 16, tensor.shape[2])
    out_view[:, :, 0].copy_(gate_view)
    out_view[:, :, 1].copy_(up_view)
    return out.contiguous()


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
    gate_up_clamp: float | None = None,
    activation_clamp: float | None = None,
) -> TransformedMegaWeights:
    """bf16 (or pre-quantized) weights → NVFP4 + swizzled-SF mega layout."""
    import torch

    _require_cutedsl_paths()
    from moe_nvfp4_swapab.mega_runner import _stack_byte_reinterpretable_tensors

    norm_const = _resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    fc1_out = 2 * intermediate_size
    num_experts = weights.w13.shape[0]

    logical_w13_shape = (num_experts, fc1_out, hidden_size)
    logical_w2_shape = (num_experts, hidden_size, intermediate_size)
    packed_w13_shape = (num_experts, fc1_out, hidden_size // 2)
    packed_w2_shape = (num_experts, hidden_size, intermediate_size // 2)

    if weights.w13_scale is not None and weights.w2_scale is not None:
        if (
            weights.w13.shape == packed_w13_shape
            and weights.w2.shape == packed_w2_shape
        ):
            if not _is_packed_nvfp4_weight(weights.w13) or not _is_packed_nvfp4_weight(
                weights.w2
            ):
                raise ValueError(
                    "packed NVFP4 weights must be torch.uint8 or torch.float4_e2m1fn_x2"
                )
        elif (
            weights.w13.shape != logical_w13_shape
            or weights.w2.shape != logical_w2_shape
        ):
            raise ValueError(
                "pre-quantized w13/w2 must have packed shapes "
                f"{packed_w13_shape} / {packed_w2_shape} or legacy logical "
                f"shapes {logical_w13_shape} / {logical_w2_shape}; got "
                f"{tuple(weights.w13.shape)} / {tuple(weights.w2.shape)}"
            )
        expected_w13_scale_shape = (
            num_experts,
            fc1_out,
            hidden_size // 16,
        )
        expected_w2_scale_shape = (
            num_experts,
            hidden_size,
            intermediate_size // 16,
        )
        if weights.w13_scale.shape != expected_w13_scale_shape:
            raise ValueError(
                f"w13_scale must have shape {expected_w13_scale_shape}, "
                f"got {tuple(weights.w13_scale.shape)}"
            )
        if weights.w2_scale.shape != expected_w2_scale_shape:
            raise ValueError(
                f"w2_scale must have shape {expected_w2_scale_shape}, "
                f"got {tuple(weights.w2_scale.shape)}"
            )
        w13 = _interleave_gate_up_16(weights.w13, intermediate_size=intermediate_size)
        w13_scale = _interleave_gate_up_16(
            weights.w13_scale, intermediate_size=intermediate_size
        )
        # Keep the transpose as a view so the kernel's K axis remains stride-1.
        # Materializing the logical (E, K, N) view would make N stride-1.
        fc1_weight = _as_fp4_weight(w13.transpose(1, 2))
        fc2_weight = _as_fp4_weight(weights.w2.transpose(1, 2))
        fc1_sf_swizzled = [
            _swizzle_expert_scales(w13_scale[e]) for e in range(num_experts)
        ]
        fc2_sf_swizzled = [
            _swizzle_expert_scales(weights.w2_scale[e]) for e in range(num_experts)
        ]
    else:
        fc1_q_parts = []
        fc1_sf_parts = []
        fc2_q_parts = []
        fc2_sf_parts = []
        if weights.w13.shape != logical_w13_shape:
            raise ValueError(
                f"w13 must have shape {logical_w13_shape}, "
                f"got {tuple(weights.w13.shape)}"
            )
        if weights.w2.shape != logical_w2_shape:
            raise ValueError(
                f"w2 must have shape {logical_w2_shape}, got {tuple(weights.w2.shape)}"
            )
        w13 = _interleave_gate_up_16(weights.w13, intermediate_size=intermediate_size)
        for expert in range(num_experts):
            fc1_q, fc1_sf = _quantize_expert_weights(
                w13[expert],
                norm_const=norm_const,
            )
            fc2_q, fc2_sf = _quantize_expert_weights(
                weights.w2[expert],
                norm_const=norm_const,
            )
            fc1_q_parts.append(fc1_q.transpose(0, 1).contiguous())
            fc1_sf_parts.append(fc1_sf)
            fc2_q_parts.append(fc2_q.transpose(0, 1).contiguous())
            fc2_sf_parts.append(fc2_sf)

        fc1_weight = torch.stack(fc1_q_parts, dim=0)
        fc2_weight = torch.stack(fc2_q_parts, dim=0)
        fc1_sf_swizzled = [_swizzle_expert_scales(sf) for sf in fc1_sf_parts]
        fc2_sf_swizzled = [_swizzle_expert_scales(sf) for sf in fc2_sf_parts]

    fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
    fc2_flat_sf_size = fc2_sf_swizzled[0].numel()
    fc1_weight_sf = _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0).view(
        num_experts, fc1_flat_sf_size
    )
    fc2_weight_sf = _stack_byte_reinterpretable_tensors(fc2_sf_swizzled, dim=0).view(
        num_experts, fc2_flat_sf_size
    )

    return (fc1_weight, fc1_weight_sf), (fc2_weight, fc2_weight_sf)


def _nvfp4_swizzled_flat_sf_size(rows: int, cols: int) -> int:
    import torch

    _require_cutedsl_paths()
    from moe_nvfp4_swapab.runner_common import to_blocked

    plain = torch.zeros(rows, cols, dtype=torch.float32)
    return to_blocked(plain).numel()


def _nvfp4_kernel_weight_dtype() -> "torch.dtype":
    import torch

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None:
        return fp4_dtype
    return torch.uint8


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    """One-time check for kernel-ready NVFP4 weights (``preprocess_weights=False``)."""
    import torch

    from .....core.validation.common import MoEEpConfigError
    from ..weight_validation import (
        check_transformed_mega_weights_structure,
        check_transformed_weight_pair,
    )

    if world_size <= 0:
        raise MoEEpConfigError(f"world_size must be positive, got {world_size}")
    if num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
        )

    local_experts = num_experts // world_size
    i_down = intermediate_size // 2
    fc1_out = 2 * intermediate_size
    weight_dtype = _nvfp4_kernel_weight_dtype()

    _require_cutedsl_paths()
    from common.megamoe_constants import Nvfp4BlockSize
    from moe_nvfp4_swapab.runner_common import ceil_div

    hidden_sf_cols = ceil_div(hidden_size, Nvfp4BlockSize)
    intermediate_sf_cols = ceil_div(intermediate_size, Nvfp4BlockSize)
    fc1_flat_sf = _nvfp4_swizzled_flat_sf_size(fc1_out, hidden_sf_cols)
    fc2_flat_sf = _nvfp4_swizzled_flat_sf_size(hidden_size, intermediate_sf_cols)

    check_transformed_mega_weights_structure(transformed)
    check_transformed_weight_pair(
        transformed[0],
        label="fc1",
        num_local_experts=local_experts,
        weight_dtype=weight_dtype,
        expected_weight_shape=(local_experts, hidden_size // 2, fc1_out),
        scale_dtype=torch.uint8,
        expected_scale_shape=(local_experts, fc1_flat_sf),
    )
    check_transformed_weight_pair(
        transformed[1],
        label="fc2",
        num_local_experts=local_experts,
        weight_dtype=weight_dtype,
        expected_weight_shape=(local_experts, i_down, hidden_size),
        scale_dtype=torch.uint8,
        expected_scale_shape=(local_experts, fc2_flat_sf),
    )


__all__ = [
    "MoEWeightPack",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
