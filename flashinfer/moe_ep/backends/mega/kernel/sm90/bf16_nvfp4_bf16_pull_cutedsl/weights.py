"""Mega-path NVFP4 (W4A16) weight preprocessing for the SM90 pull-style kernel.

Each leg is the per-tensor ABI 4-tuple ``(weight, weight_sf,
activation_dequant_scale, weight_dequant_scale)``: the augmented row-pair
byte tensor ``(E, N/2, K/128 * 144)`` (see ``..common.nvfp4``), the unit E8M0
SF placeholder plane, ``ones(1)``, and the FP32 per-expert global scale
``(E,)`` that the epilogue applies.  FC1 gate/up rows are interleaved in
blocks of 8 (``Fp8GateUpInterleave``), together with their block scales,
before the pairs are formed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ......weights import MoEWeightPack, PrequantizedMoEWeights
from ..common.nvfp4 import (
    NVFP4_BLOCK_K,
    W4A16_PAIR_TILE_BYTES,
    W4A16_TILE_K,
    augment_w4a16,
    packed_bytes,
    quantize_nvfp4,
)
from ..fp8_fp8_bf16_pull_cutedsl.weights import (
    TransformedMegaWeights,
    _interleave_gate_up_8,
    _swizzle_unit_e8m0_sf,
    _swizzled_flat_e8m0_size,
)

if TYPE_CHECKING:
    import torch

_KERNEL = "sm90_bf16_nvfp4_bf16_pull_cutedsl"


def _alpha(
    alpha: "torch.Tensor | None", num_experts: int, device, name: str
) -> "torch.Tensor":
    import torch

    from ......core.validation.common import MoEEpConfigError

    if alpha is None:
        return torch.ones(num_experts, dtype=torch.float32, device=device)
    if (
        not isinstance(alpha, torch.Tensor)
        or alpha.dtype != torch.float32
        or tuple(alpha.shape) != (num_experts,)
    ):
        raise MoEEpConfigError(
            f"{_KERNEL} {name} must be an FP32 [local_experts] tensor"
        )
    return alpha.to(device).clone()


def _nvfp4_legs(weights, *, intermediate_size, hidden_size):
    """(packed, scale bytes, alpha) per leg from an NVFP4 or BF16 pack."""
    import torch

    from ......core.validation.common import MoEEpConfigError

    num_experts = weights.w13.shape[0]
    legs = (
        ("w13", (num_experts, 2 * intermediate_size, hidden_size)),
        ("w2", (num_experts, hidden_size, intermediate_size)),
    )
    out = []
    for name, (e, n, k) in legs:
        data = getattr(weights, name)
        if isinstance(weights, PrequantizedMoEWeights):
            scale = getattr(weights, f"{name}_scale")
            try:
                packed = packed_bytes(data)
            except ValueError as exc:
                raise MoEEpConfigError(f"{_KERNEL} {name}: {exc}") from exc
            if tuple(packed.shape) != (e, n, k // 2):
                raise MoEEpConfigError(
                    f"{_KERNEL} packed {name} must have shape {(e, n, k // 2)}, "
                    f"got {tuple(packed.shape)}"
                )
            if scale.dtype not in (torch.float8_e4m3fn, torch.uint8) or tuple(
                scale.shape
            ) != (e, n, k // NVFP4_BLOCK_K):
                raise MoEEpConfigError(
                    f"{_KERNEL} {name}_scale must be linear E4M3 (or uint8) of "
                    f"shape {(e, n, k // NVFP4_BLOCK_K)}, got {scale.dtype} "
                    f"{tuple(scale.shape)}"
                )
            if scale.device != packed.device:
                raise MoEEpConfigError(
                    f"{_KERNEL} {name} and its scale must share a device"
                )
            out.append((packed, scale.view(torch.uint8), None))
        else:
            if tuple(data.shape) != (e, n, k):
                raise MoEEpConfigError(
                    f"{_KERNEL} {name} must have shape {(e, n, k)}, got {tuple(data.shape)}"
                )
            out.append(quantize_nvfp4(data))
    return out


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
    fc1_alpha: "torch.Tensor | None" = None,
    fc2_alpha: "torch.Tensor | None" = None,
) -> TransformedMegaWeights:
    """NVFP4 pack (or canonical BF16/FP16/FP32, quantized here) -> W4A16 legs.

    ``fc1_alpha`` / ``fc2_alpha``: FP32 per-expert global scales.  For a BF16
    pack the quantizer's own global scale is folded in on top.
    """
    import torch

    from ......core.validation.common import MoEEpConfigError

    if hidden_size % W4A16_TILE_K or intermediate_size % W4A16_TILE_K:
        raise MoEEpConfigError(
            f"{_KERNEL} needs hidden_size and intermediate_size to be multiples of "
            f"{W4A16_TILE_K}; got {hidden_size}, {intermediate_size}"
        )
    num_experts = weights.w13.shape[0]
    device = weights.w13.device
    (p13, s13, q13), (p2, s2, q2) = _nvfp4_legs(
        weights, intermediate_size=intermediate_size, hidden_size=hidden_size
    )
    alpha1 = _alpha(fc1_alpha, num_experts, device, "fc1_alpha")
    alpha2 = _alpha(fc2_alpha, num_experts, device, "fc2_alpha")
    if q13 is not None:
        alpha1 = alpha1 * q13
        alpha2 = alpha2 * q2

    # Gate/up interleave is a row permutation: move each row's scales with it.
    fc1_out = 2 * intermediate_size
    p13 = _interleave_gate_up_8(p13, intermediate_size=fc1_out)
    s13 = _interleave_gate_up_8(s13, intermediate_size=fc1_out)
    try:
        a13 = augment_w4a16(p13, s13)
        a2 = augment_w4a16(p2, s2)
    except ValueError as exc:
        raise MoEEpConfigError(f"{_KERNEL}: {exc}") from exc
    ones_1 = torch.ones(1, dtype=torch.float32, device=device)
    return (
        (
            a13,
            _swizzle_unit_e8m0_sf(num_experts, fc1_out, hidden_size // 32, device),
            ones_1,
            alpha1,
        ),
        (
            a2,
            _swizzle_unit_e8m0_sf(
                num_experts, hidden_size, intermediate_size // 32, device
            ),
            ones_1.clone(),
            alpha2,
        ),
    )


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    """One-time check for kernel-ready W4A16 legs (``preprocess_weights=False``)."""
    import torch

    from ......core.validation.common import MoEEpConfigError
    from ...weight_validation import check_transformed_weight_pair
    from ..fp8_fp8_bf16_pull_cutedsl.weights import _check_leg_structure

    if world_size <= 0 or num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
        )
    _check_leg_structure(transformed)
    local = num_experts // world_size
    fc1_out = 2 * intermediate_size
    for idx, (label, rows, k) in enumerate(
        (("fc1", fc1_out, hidden_size), ("fc2", hidden_size, intermediate_size))
    ):
        weight, sf, act_scale, alpha = transformed[idx]
        check_transformed_weight_pair(
            (weight, sf),
            label=label,
            num_local_experts=local,
            weight_dtype=torch.uint8,
            expected_weight_shape=(
                local,
                rows // 2,
                k // W4A16_TILE_K * W4A16_PAIR_TILE_BYTES,
            ),
            scale_dtype=torch.uint8,
            expected_scale_shape=(local, _swizzled_flat_e8m0_size(rows, k // 32)),
        )
        if not weight.is_contiguous():
            raise MoEEpConfigError(
                f"transformed_weights {label} weight must be contiguous"
            )
        if (
            not isinstance(act_scale, torch.Tensor)
            or tuple(act_scale.shape) != (1,)
            or act_scale.dtype != torch.float32
            or not bool(torch.all(act_scale == 1))
        ):
            raise MoEEpConfigError(
                f"transformed_weights {label} activation_dequant_scale must be "
                "float32 ones of shape (1,)"
            )
        if (
            not isinstance(alpha, torch.Tensor)
            or tuple(alpha.shape) != (local,)
            or alpha.dtype != torch.float32
        ):
            raise MoEEpConfigError(
                f"transformed_weights {label} weight_dequant_scale (alpha) must be "
                f"float32 of shape ({local},)"
            )


__all__ = [
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
