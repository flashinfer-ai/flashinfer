"""Mega-path MXFP8 weight preprocessing for CuTeDSL MegaMoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

from ......weights import MoEWeightPack, PrequantizedMoEWeights
from ..common import mxfp8_weight_utils
from ..common.mxfp8_weight_utils import (
    as_mxfp8_scale as _as_mxfp8_scale,
    interleave_gate_up,
    mxfp8_data_dtype as _mxfp8_data_dtype,
)

if TYPE_CHECKING:
    import torch

TransformedMegaWeights = Tuple[
    Tuple["torch.Tensor", "torch.Tensor"],
    Tuple["torch.Tensor", "torch.Tensor"],
]

Mxfp8Kind = Literal["mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"]
_quantize_mxfp8_weight_k_major = mxfp8_weight_utils.quantize_mxfp8_weight_k_major


def _fc1_weight_from_w13(
    w13: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    """(E, 2*I, H) gate||up bf16 -> (E, H, 2*I) kernel FC1 layout."""
    _, two_i, _ = w13.shape
    i = intermediate_size
    if two_i != 2 * i:
        raise ValueError(
            f"expected w13 with {2 * i} rows (gate||up), got shape {tuple(w13.shape)}"
        )
    return (
        _interleave_gate_up_32(w13, intermediate_size=2 * i)
        .transpose(1, 2)
        .contiguous()
    )


def _interleave_gate_up_32(
    tensor: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification import (
        Mxfp8BlockSize,
    )

    return interleave_gate_up(
        tensor,
        intermediate_size=intermediate_size,
        block_size=Mxfp8BlockSize,
        kernel_name="MXFP8 MegaMOE",
    )


def _fc1_kernel_weight_from_canonical_mxfp8(
    w13: "torch.Tensor", *, intermediate_size: int
) -> "torch.Tensor":
    return (
        _interleave_gate_up_32(w13, intermediate_size=2 * intermediate_size)
        .transpose(1, 2)
        .contiguous()
    )


def _is_mxfp8_weight(weight: "torch.Tensor", *, kind: Mxfp8Kind) -> bool:
    return weight.dtype == _mxfp8_data_dtype(kind)


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
    kind: Mxfp8Kind = "mxfp8_e4m3",
    gate_up_clamp: float | None = None,
    activation_clamp: float | None = None,
    kernel_variant: Literal["inference", "genphase"] = "inference",
) -> TransformedMegaWeights:
    """Prepare canonical weights for standard or GenPhase.

    FC1 w13 is [E, 2I, H], with gate rows followed by up rows; FC2 w2 is
    [E, H, I]. Prequantized MXFP8 inputs retain those shapes, with plain
    E8M0 scales [E, 2I, H/32] and [E, H, I/32]. Pre-interleaved kernel-layout
    FC1 inputs from the retired legacy implementation are not accepted.
    """
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import (
        preprocess_block_scaled_weights,
        preprocess_prequantized_block_scaled_weights,
    )

    if kernel_variant not in ("inference", "genphase"):
        raise ValueError(f"unsupported kernel_variant {kernel_variant!r}")
    del gate_up_clamp, activation_clamp
    if isinstance(weights, PrequantizedMoEWeights):
        return preprocess_prequantized_block_scaled_weights(
            weights.w13,
            weights.w2,
            _as_mxfp8_scale(weights.w13_scale),
            _as_mxfp8_scale(weights.w2_scale),
            quant_kind=kind,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    if weights.w13.ndim != 3 or weights.w13.shape[1:] != (
        2 * intermediate_size,
        hidden_size,
    ):
        raise ValueError("w13 must have canonical shape [E, 2I, H].")
    return preprocess_block_scaled_weights(
        weights.w13,
        weights.w2,
        quant_kind=kind,
        intermediate_size=intermediate_size,
    )


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    kind: Mxfp8Kind = "mxfp8_e4m3",
    world_size: int,
    num_experts: int,
) -> None:
    """Validate the sources-backed weight layout used by the public selector."""
    import torch

    from ......core.validation.common import MoEEpConfigError
    from ...weight_validation import check_transformed_mega_weights_structure
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import (
        Sm100BlockScaledMoeConfig,
        validate_standard_weights,
    )

    check_transformed_mega_weights_structure(transformed)
    try:
        config = Sm100BlockScaledMoeConfig(
            num_total_experts=num_experts,
            max_tokens_per_rank=1,
            num_topk=1,
            hidden=hidden_size,
            intermediate=intermediate_size,
            rank=0,
            world_size=world_size,
            quant_kind=kind,
        )
        device = getattr(transformed[0][0], "device", torch.device("cpu"))
        validate_standard_weights(transformed, config, device)
    except ValueError as exc:
        raise MoEEpConfigError(str(exc)) from exc


__all__ = [
    "MoEWeightPack",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
