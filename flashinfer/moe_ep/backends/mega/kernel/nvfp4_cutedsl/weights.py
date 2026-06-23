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
    import cutedsl_nvfp4_mega_moe_front_end  # noqa: F401


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

    if weights.w13.shape != (num_experts, fc1_out, hidden_size):
        raise ValueError(
            f"w13 must have shape ({num_experts}, {fc1_out}, {hidden_size}), "
            f"got {tuple(weights.w13.shape)}"
        )
    if weights.w2.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError(
            f"w2 must have shape ({num_experts}, {hidden_size}, {intermediate_size}), "
            f"got {tuple(weights.w2.shape)}"
        )

    if weights.w13_scale is not None and weights.w2_scale is not None:
        fc1_weight = weights.w13.transpose(1, 2).contiguous()
        fc2_weight = weights.w2.transpose(1, 2).contiguous()
        fc1_sf_swizzled = [
            _swizzle_expert_scales(weights.w13_scale[e]) for e in range(num_experts)
        ]
        fc2_sf_swizzled = [
            _swizzle_expert_scales(weights.w2_scale[e]) for e in range(num_experts)
        ]
    else:
        fc1_q_parts = []
        fc1_sf_parts = []
        fc2_q_parts = []
        fc2_sf_parts = []
        for expert in range(num_experts):
            fc1_q, fc1_sf = _quantize_expert_weights(
                weights.w13[expert],
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


__all__ = ["MoEWeightPack", "TransformedMegaWeights", "preprocess_mega_weights"]
