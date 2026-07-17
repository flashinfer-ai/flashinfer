"""Weight preprocessing for the SM90 push FP8 mega-MoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from .....core.validation.common import MoEEpConfigError
from .....weights import MoEWeightPack

if TYPE_CHECKING:
    from .....kernel_src.sm90_push_megamoe import Sm90PushWeights

    TransformedMegaWeights: TypeAlias = Sm90PushWeights


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from .....kernel_src.sm90_push_megamoe import Sm90PushWeights

        return Sm90PushWeights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_transformed_mega_weights(
    transformed_weights: object,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
    fuse_fc1_epilogue: bool,
) -> None:
    import torch

    from .....kernel_src.sm90_push_megamoe import Sm90PushWeights

    if not isinstance(transformed_weights, Sm90PushWeights):
        raise MoEEpConfigError(
            "sm90_push_fp8 transformed weights must be Sm90PushWeights, got "
            f"{type(transformed_weights).__name__}"
        )
    expected_w13 = (num_local_experts, 2 * intermediate_size, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    expected_w13_sf = (
        num_local_experts,
        2 * intermediate_size // 128,
        hidden_size // 128,
    )
    expected_w2_sf = (
        num_local_experts,
        hidden_size // 128,
        intermediate_size // 128,
    )
    expected = (
        ("w13_fp8", transformed_weights.w13_fp8, expected_w13, torch.float8_e4m3fn),
        ("w13_sf", transformed_weights.w13_sf, expected_w13_sf, torch.float32),
        ("w2_fp8", transformed_weights.w2_fp8, expected_w2, torch.float8_e4m3fn),
        ("w2_sf", transformed_weights.w2_sf, expected_w2_sf, torch.float32),
    )
    device = transformed_weights.w13_fp8.device
    if not transformed_weights.w13_fp8.is_cuda:
        raise MoEEpConfigError("sm90_push_fp8 transformed weights must be CUDA tensors")
    for name, tensor, shape, dtype in expected:
        if tuple(tensor.shape) != shape:
            raise MoEEpConfigError(
                f"sm90_push_fp8 {name} must have shape {shape}, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise MoEEpConfigError(
                f"sm90_push_fp8 {name} must have dtype {dtype}, got {tensor.dtype}"
            )
        if tensor.device != device:
            raise MoEEpConfigError(
                f"sm90_push_fp8 {name} must be on {device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"sm90_push_fp8 {name} must be contiguous")
    if transformed_weights.w13_interleaved != fuse_fc1_epilogue:
        raise MoEEpConfigError(
            "sm90_push_fp8 weight layout does not match fuse_fc1_epilogue: "
            f"w13_interleaved={transformed_weights.w13_interleaved}, "
            f"fuse_fc1_epilogue={fuse_fc1_epilogue}"
        )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
    fuse_fc1_epilogue: bool,
) -> Any:
    import torch

    from .....kernel_src.sm90_push_megamoe import make_sm90_push_weights

    if not isinstance(weights, MoEWeightPack):
        raise MoEEpConfigError(
            f"sm90_push_fp8 weights must be MoEWeightPack, got {type(weights).__name__}"
        )
    if weights.w13_scale is not None or weights.w2_scale is not None:
        raise MoEEpConfigError(
            "sm90_push_fp8 preprocessing accepts canonical bf16 weights only; "
            "pass Sm90PushWeights through MegaConfig.transformed_weights to reuse "
            "an already transformed bundle"
        )
    expected_w13 = (num_local_experts, 2 * intermediate_size, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    for name, tensor, shape in (
        ("w13", weights.w13, expected_w13),
        ("w2", weights.w2, expected_w2),
    ):
        if tuple(tensor.shape) != shape:
            raise MoEEpConfigError(
                f"sm90_push_fp8 {name} must have shape {shape}, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"sm90_push_fp8 {name} must be torch.bfloat16, got {tensor.dtype}"
            )
        if not tensor.is_cuda:
            raise MoEEpConfigError(f"sm90_push_fp8 {name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"sm90_push_fp8 {name} must be contiguous")
    if weights.w13.device != weights.w2.device:
        raise MoEEpConfigError(
            "sm90_push_fp8 w13 and w2 must be on the same CUDA device"
        )
    transformed = make_sm90_push_weights(
        weights.w13,
        weights.w2,
        interleave_gate_up=fuse_fc1_epilogue,
    )
    validate_transformed_mega_weights(
        transformed,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        num_local_experts=num_local_experts,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    return transformed


__all__ = [
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
