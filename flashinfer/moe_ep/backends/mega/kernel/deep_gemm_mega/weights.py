"""Mega-path weight preprocessing."""

from __future__ import annotations

from typing import Tuple

import torch

from .....weights import MoEWeightPack

TransformedMegaWeights = Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
    return (sf.to(torch.int32) << 23).view(torch.float32)


def _weight_sf_to_float(w_sf: torch.Tensor) -> torch.Tensor:
    if w_sf.dtype == torch.uint8:
        return _ue8m0_uint8_to_float(w_sf)
    return w_sf.contiguous()


def _as_fp4_weight(w: torch.Tensor) -> torch.Tensor:
    if w.dtype in (torch.uint8, torch.int8):
        return w.view(torch.int8).contiguous()
    raise TypeError(
        "pre-quantized MoEWeightPack weights must be fp4 (torch.int8 or torch.uint8); "
        f"got {w.dtype}. Pass w13_scale/w2_scale together, or supply bf16 weights "
        "without scales to quantize during preprocess."
    )


def _quantize_grouped_fp4(
    bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from deep_gemm.utils import per_token_cast_to_fp4

    num_groups, n, k = bf16.shape
    w = torch.empty((num_groups, n, k // 2), device=bf16.device, dtype=torch.int8)
    w_sf = torch.empty(
        (num_groups, n, k // 32), device=bf16.device, dtype=torch.float32
    )
    for i in range(num_groups):
        w[i], w_sf[i] = per_token_cast_to_fp4(bf16[i], use_ue8m0=True, gran_k=32)
    return w, w_sf


def _transform_weight_sf(
    w: torch.Tensor,
    w_sf: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm

    num_groups, n, k_fp4 = w.shape
    k = k_fp4 * 2
    return w, deep_gemm.transform_sf_into_required_layout(
        w_sf, n, k, (1, 32), num_groups
    )


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """User fp4+scale (or bf16) weights → layout expected by ``fp8_fp4_mega_moe``."""
    import deep_gemm

    if weights.w13_scale is None or weights.w2_scale is None:
        l1 = _transform_weight_sf(*_quantize_grouped_fp4(weights.w13))
        l2 = _transform_weight_sf(*_quantize_grouped_fp4(weights.w2))
    else:
        l1 = _transform_weight_sf(
            _as_fp4_weight(weights.w13),
            _weight_sf_to_float(weights.w13_scale),
        )
        l2 = _transform_weight_sf(
            _as_fp4_weight(weights.w2),
            _weight_sf_to_float(weights.w2_scale),
        )

    return deep_gemm.transform_weights_for_mega_moe(l1, l2)


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    """One-time check for kernel-ready DeepGEMM fp4 weights (``preprocess_weights=False``)."""
    import torch

    from .....core.validation.common import MoEEpConfigError

    if world_size <= 0:
        raise MoEEpConfigError(f"world_size must be positive, got {world_size}")
    if num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
        )

    local_experts = num_experts // world_size
    fc1_out = 2 * intermediate_size
    expected_fc1_w = (local_experts, fc1_out, hidden_size // 2)
    expected_fc2_w = (local_experts, hidden_size, intermediate_size // 2)

    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise MoEEpConfigError(
            f"transformed_weights must be a 2-tuple (fc1, fc2), got {type(transformed).__name__}"
        )

    for label, pair, expected_shape in (
        ("fc1", transformed[0], expected_fc1_w),
        ("fc2", transformed[1], expected_fc2_w),
    ):
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        weight, scale = pair
        if not isinstance(weight, torch.Tensor):
            raise MoEEpConfigError(
                f"transformed_weights {label} weight must be a torch.Tensor, "
                f"got {type(weight).__name__}"
            )
        if not isinstance(scale, torch.Tensor):
            raise MoEEpConfigError(
                f"transformed_weights {label} scale must be a torch.Tensor, "
                f"got {type(scale).__name__}"
            )
        if weight.dtype != torch.int8:
            raise MoEEpConfigError(
                f"transformed_weights {label} weight must be torch.int8 (NVFP4), "
                f"got {weight.dtype}"
            )
        if weight.shape != expected_shape:
            raise MoEEpConfigError(
                f"transformed_weights {label} weight must have shape "
                f"{expected_shape}, got {tuple(weight.shape)}"
            )
        if weight.shape[0] != local_experts:
            raise MoEEpConfigError(
                f"transformed_weights {label} leading dim ({weight.shape[0]}) must "
                f"match num_experts // world_size ({local_experts})"
            )
        if scale.shape[0] != local_experts:
            raise MoEEpConfigError(
                f"transformed_weights {label} scale leading dim ({scale.shape[0]}) "
                f"must match num_experts // world_size ({local_experts})"
            )


__all__ = [
    "MoEWeightPack",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
