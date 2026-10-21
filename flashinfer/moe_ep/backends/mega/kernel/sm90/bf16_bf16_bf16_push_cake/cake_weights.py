"""Weight preprocessing for the SM90 native BF16 push mega-MoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from ......core.validation.common import MoEEpConfigError
from ......weights import MoEWeightPack

if TYPE_CHECKING:
    from ......kernel_src.sm90.cake_bf16_megamoe import Sm90CakeBf16Weights

    TransformedMegaWeights: TypeAlias = Sm90CakeBf16Weights


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from ......kernel_src.sm90.cake_bf16_megamoe import Sm90CakeBf16Weights

        return Sm90CakeBf16Weights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_transformed_mega_weights(
    transformed_weights: object,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
) -> None:
    import torch

    from ......kernel_src.sm90.cake_bf16_megamoe import Sm90CakeBf16Weights

    if not isinstance(transformed_weights, Sm90CakeBf16Weights):
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake transformed weights must be Sm90CakeBf16Weights, got "
            f"{type(transformed_weights).__name__}"
        )
    if (
        transformed_weights.num_local_experts != num_local_experts
        or transformed_weights.hidden_size != hidden_size
        or transformed_weights.intermediate_size != intermediate_size
    ):
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake transformed weights describe "
            f"(E={transformed_weights.num_local_experts}, H={transformed_weights.hidden_size}, "
            f"I={transformed_weights.intermediate_size}); the layer needs "
            f"(E={num_local_experts}, H={hidden_size}, I={intermediate_size})"
        )
    for name, tensor in (
        ("w13", transformed_weights.w13),
        ("w2", transformed_weights.w2),
    ):
        if tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"sm90_bf16_bf16_bf16_push_cake {name} must be torch.bfloat16, got {tensor.dtype}"
            )
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise MoEEpConfigError(
                f"sm90_bf16_bf16_bf16_push_cake {name} must be a contiguous CUDA tensor"
            )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
) -> Any:
    """Interleave canonical bf16 ``w13`` gate/up rows for the fused FC1 kernel.

    Accepts canonical bf16 weights only: this backend never quantizes, so a
    pre-quantized :class:`MoEWeightPack` (with scales) is rejected.
    """
    import torch

    from ......kernel_src.sm90.cake_bf16_megamoe import make_sm90_cake_bf16_weights

    if not isinstance(weights, MoEWeightPack):
        raise MoEEpConfigError(
            f"sm90_bf16_bf16_bf16_push_cake weights must be MoEWeightPack, got {type(weights).__name__}"
        )
    if (
        getattr(weights, "w13_scale", None) is not None
        or getattr(weights, "w2_scale", None) is not None
    ):
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake is a native BF16 backend and accepts canonical bf16 "
            "weights only (no scale planes)"
        )
    expected_w13 = (num_local_experts, 2 * intermediate_size, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    for name, tensor, shape in (
        ("w13", weights.w13, expected_w13),
        ("w2", weights.w2, expected_w2),
    ):
        if tuple(tensor.shape) != shape:
            raise MoEEpConfigError(
                f"sm90_bf16_bf16_bf16_push_cake {name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"sm90_bf16_bf16_bf16_push_cake {name} must be torch.bfloat16, got {tensor.dtype}"
            )
        if not tensor.is_cuda:
            raise MoEEpConfigError(
                f"sm90_bf16_bf16_bf16_push_cake {name} must be a CUDA tensor"
            )
    if weights.w13.device != weights.w2.device:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake w13 and w2 must be on the same CUDA device"
        )
    return make_sm90_cake_bf16_weights(
        weights.w13.contiguous(), weights.w2.contiguous()
    )
