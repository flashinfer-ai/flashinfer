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
    import torch

    if isinstance(weights, PrequantizedMoEWeights):
        raise MoEEpConfigError(
            "pre-quantized weights are not supported by the "
            "sm107_nvfp4_nvfp4_bf16_cutedsl backend yet; pass canonical "
            "bf16/fp32 weights."
        )

    # Backend talks only to the next_cutedsl_megamoe shim (never src/ directly).
    from ......kernel_src.next_cutedsl_megamoe import (
        interleave_gate_up_16,
        quantize_nvfp4_block16,
        to_blocked,
    )

    w13, w2 = weights.w13, weights.w2
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

    # FC1: interleave the gate‖up halves into 16-row pair stripes, quantize
    # along K (hidden, the trailing dim; also the fp4 pack axis), then expose
    # the kernel's logical (E, hidden/2, 2I) view with packed-hidden stride-1.
    w13_interleaved = interleave_gate_up_16(
        w13.to(torch.float32), intermediate_size=intermediate_size
    ).contiguous()
    fc1_q, fc1_sf = quantize_nvfp4_block16(w13_interleaved)
    fc1_weight = fc1_q.permute(0, 2, 1)

    # FC2: quantize along K (intermediate, the trailing dim of canonical w2),
    # then expose the logical (E, intermediate/2, hidden) view.
    fc2_q, fc2_sf = quantize_nvfp4_block16(w2.to(torch.float32).contiguous())
    fc2_weight = fc2_q.permute(0, 2, 1)

    fc1_weight_sf = torch.stack(
        [to_blocked(fc1_sf[e].view(torch.uint8)) for e in range(num_local_experts)]
    ).view(torch.float8_e4m3fn)
    fc2_weight_sf = torch.stack(
        [to_blocked(fc2_sf[e].view(torch.uint8)) for e in range(num_local_experts)]
    ).view(torch.float8_e4m3fn)

    return ((fc1_weight, fc1_weight_sf), (fc2_weight, fc2_weight_sf))


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

    from ......kernel_src.next_cutedsl_megamoe import (
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
