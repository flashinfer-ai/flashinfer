"""Weight transform + validation for the SM107 mxfp8 block-scaled mega kernel.

Kernel weight layout (see the drop's ``generate_inputs`` contract):

- FC1: logical ``(E, hidden, 2*intermediate)`` with hidden stride-1 and the N
  axis in 16-row gate/up pair stripes; flat atom-swizzled E8M0 SF plane of
  ``round_up(2*intermediate, 128) * round_up(hidden/32, 4)`` per expert.
- FC2: logical ``(E, intermediate, hidden)`` with intermediate stride-1; flat
  SF plane of ``round_up(hidden, 128) * round_up(intermediate/32, 4)``.
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
from .config import Sm107Mxfp8Kind

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


def _data_dtype(kind: Sm107Mxfp8Kind) -> "torch.dtype":
    import torch

    if kind not in ("mxfp8_e4m3", "mxfp8_e5m2"):
        raise MoEEpConfigError(f"unsupported MXFP8 kind {kind!r}")
    return torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    kind: Sm107Mxfp8Kind = "mxfp8_e4m3",
) -> TransformedMegaWeights:
    """Canonical bf16 ``w13``/``w2`` -> SM107 mxfp8 kernel layout.

    Pre-quantized packs are not supported yet (the kernel-layout + swizzled-SF
    import path can be added when a producer exists).
    """
    if isinstance(weights, PrequantizedMoEWeights):
        raise MoEEpConfigError(
            "pre-quantized weights are not supported by the "
            "sm107_mxfp8_mxfp8_bf16_cutedsl backend yet; pass canonical "
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

    _data_dtype(kind)
    return preprocess_block_scaled_weights(
        w13, w2, quant_kind=kind, intermediate_size=intermediate_size
    )


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    kind: Sm107Mxfp8Kind,
    world_size: int,
    num_experts: int,
) -> None:
    """Structure/shape/dtype checks for user-supplied transformed weights."""
    import torch

    from ......kernel_src.sm107.next_cutedsl_megamoe import (
        Mxfp8BlockSize,
        swizzled_flat_sf_size,
    )

    check_transformed_mega_weights_structure(transformed)
    num_local_experts = num_experts // world_size
    fc1_out = 2 * intermediate_size
    check_transformed_weight_pair(
        transformed[0],
        label="fc1",
        num_local_experts=num_local_experts,
        weight_dtype=_data_dtype(kind),
        expected_weight_shape=(num_local_experts, hidden_size, fc1_out),
        scale_dtype=torch.float8_e8m0fnu,
        expected_scale_shape=(
            num_local_experts,
            swizzled_flat_sf_size(fc1_out, hidden_size // Mxfp8BlockSize),
        ),
    )
    check_transformed_weight_pair(
        transformed[1],
        label="fc2",
        num_local_experts=num_local_experts,
        weight_dtype=_data_dtype(kind),
        expected_weight_shape=(num_local_experts, intermediate_size, hidden_size),
        scale_dtype=torch.float8_e8m0fnu,
        expected_scale_shape=(
            num_local_experts,
            swizzled_flat_sf_size(hidden_size, intermediate_size // Mxfp8BlockSize),
        ),
    )
    for weight, scale in transformed:
        validate_weight_layout(weight, scale, scale_dtype=torch.float8_e8m0fnu)
