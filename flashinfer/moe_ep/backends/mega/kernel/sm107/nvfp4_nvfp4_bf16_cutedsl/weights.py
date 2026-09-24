"""Weight transform + validation for the SM107 nvfp4 block-scaled mega kernel.

Kernel weight layout (see the drop's ``generate_inputs`` contract):

- FC1: logical ``(E, hidden, 2*intermediate)`` with hidden (the fp4 pack axis)
  stride-1 and the N axis in 16-row gate/up pair stripes; torch carries the
  packed ``(E, hidden/2, 2*intermediate)`` ``float4_e2m1fn_x2`` view.  Flat
  atom-swizzled FP8-E4M3 SF plane of
  ``round_up(2*intermediate, 128) * round_up(hidden/16, 4)`` per expert.
- FC2: logical ``(E, intermediate, hidden)`` with intermediate (packed)
  stride-1; flat SF plane of
  ``round_up(hidden, 128) * round_up(intermediate/16, 4)``.

Quantization uses norm_const=1.0, so the kernel's optional per-expert
fc1_alpha / fc2_alpha / fc1_norm_const scalars stay omitted (identically 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ......core.validation.common import MoEEpConfigError
from ......weights import MoEWeightPack, PrequantizedMoEWeights
from ..validation import validate_weight_layout
from ...weight_validation import (
    check_transformed_mega_weights_structure,
    check_transformed_weight_pair,
)

if TYPE_CHECKING:
    import torch

TransformedMegaWeights = Tuple[
    Tuple["torch.Tensor", "torch.Tensor"],
    Tuple["torch.Tensor", "torch.Tensor"],
]

__all__ = [
    "MoEWeightPack",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]


def _fp4_storage_dtype() -> "torch.dtype":
    import torch

    return getattr(torch, "float4_e2m1fn_x2", torch.uint8)


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """Canonical bf16 ``w13``/``w2`` -> SM107 nvfp4 kernel layout.

    Pre-quantized packs are not supported yet (the kernel-layout + swizzled-SF
    import path can be added when a producer exists).
    """
    if isinstance(weights, PrequantizedMoEWeights):
        raise MoEEpConfigError(
            "pre-quantized weights are not supported by the "
            "sm107_nvfp4_nvfp4_bf16_cutedsl backend yet; pass canonical "
            "bf16/fp32 weights."
        )

    from ......kernel_src.sm107.next_cutedsl_megamoe import (
        preprocess_block_scaled_weights,
    )

    w13, w2 = weights.w13, weights.w2
    if w13.ndim != 3 or w2.ndim != 3:
        raise MoEEpConfigError("canonical w13 and w2 must both be 3D tensors")
    num_local_experts = w13.shape[0]
    fc1_out = 2 * intermediate_size
    if tuple(w13.shape) != (num_local_experts, fc1_out, hidden_size):
        raise MoEEpConfigError(
            f"w13 shape {tuple(w13.shape)} != "
            f"({num_local_experts}, {fc1_out}, {hidden_size})"
        )
    if tuple(w2.shape) != (num_local_experts, hidden_size, intermediate_size):
        raise MoEEpConfigError(
            f"w2 shape {tuple(w2.shape)} != "
            f"({num_local_experts}, {hidden_size}, {intermediate_size})"
        )

    return preprocess_block_scaled_weights(
        w13, w2, quant_kind="nvfp4", intermediate_size=intermediate_size
    )


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    """Structure/shape/dtype checks for user-supplied transformed weights."""
    import torch

    from ......kernel_src.sm107.next_cutedsl_megamoe import (
        Nvfp4BlockSize,
        swizzled_flat_sf_size,
    )

    check_transformed_mega_weights_structure(transformed)
    num_local_experts = num_experts // world_size
    fc1_out = 2 * intermediate_size
    check_transformed_weight_pair(
        transformed[0],
        label="fc1",
        num_local_experts=num_local_experts,
        weight_dtype=_fp4_storage_dtype(),
        expected_weight_shape=(num_local_experts, hidden_size // 2, fc1_out),
        scale_dtype=torch.float8_e4m3fn,
        expected_scale_shape=(
            num_local_experts,
            swizzled_flat_sf_size(fc1_out, hidden_size // Nvfp4BlockSize),
        ),
    )
    check_transformed_weight_pair(
        transformed[1],
        label="fc2",
        num_local_experts=num_local_experts,
        weight_dtype=_fp4_storage_dtype(),
        expected_weight_shape=(num_local_experts, intermediate_size // 2, hidden_size),
        scale_dtype=torch.float8_e4m3fn,
        expected_scale_shape=(
            num_local_experts,
            swizzled_flat_sf_size(hidden_size, intermediate_size // Nvfp4BlockSize),
        ),
    )
    for weight, scale in transformed:
        validate_weight_layout(weight, scale, scale_dtype=torch.float8_e4m3fn)
