"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Weight preparation for the SM90 push BF16 mega-MoE backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from ......core.validation.common import MoEEpConfigError
from ......weights import MoEWeightPack

if TYPE_CHECKING:
    from ......kernel_src.sm90.push_style_megamoe import Sm90PushBf16Weights

    TransformedMegaWeights: TypeAlias = Sm90PushBf16Weights


def __getattr__(name: str) -> object:
    if name == "TransformedMegaWeights":
        from ......kernel_src.sm90.push_style_megamoe import Sm90PushBf16Weights

        return Sm90PushBf16Weights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_transformed_mega_weights(
    transformed_weights: object,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
    fuse_fc1_epilogue: bool = False,
) -> None:
    import torch

    from ......kernel_src.sm90.push_style_megamoe import Sm90PushBf16Weights

    if not isinstance(transformed_weights, Sm90PushBf16Weights):
        raise MoEEpConfigError(
            "sm90_push_bf16 transformed weights must be Sm90PushBf16Weights, got "
            f"{type(transformed_weights).__name__}"
        )
    expected = (
        (
            "w13",
            transformed_weights.w13,
            (num_local_experts, 2 * intermediate_size, hidden_size),
        ),
        (
            "w2",
            transformed_weights.w2,
            (num_local_experts, hidden_size, intermediate_size),
        ),
    )
    device = transformed_weights.w13.device
    if not transformed_weights.w13.is_cuda:
        raise MoEEpConfigError(
            "sm90_push_bf16 transformed weights must be CUDA tensors"
        )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise MoEEpConfigError(
                f"sm90_push_bf16 {name} must have shape {shape}, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"sm90_push_bf16 {name} must have dtype torch.bfloat16, got "
                f"{tensor.dtype}"
            )
        if tensor.device != device:
            raise MoEEpConfigError(
                f"sm90_push_bf16 {name} must be on {device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"sm90_push_bf16 {name} must be contiguous")
        required_alignment = 32 if name == "w13" and fuse_fc1_epilogue else 16
        if tensor.data_ptr() % required_alignment:
            raise MoEEpConfigError(
                f"sm90_push_bf16 {name} must be {required_alignment}-byte aligned"
            )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
    fuse_fc1_epilogue: bool = False,
) -> Any:
    import torch

    from ......kernel_src.sm90.push_style_megamoe import (
        make_sm90_push_bf16_weights,
    )

    if not isinstance(weights, MoEWeightPack):
        raise MoEEpConfigError(
            f"sm90_push_bf16 weights must be MoEWeightPack, got "
            f"{type(weights).__name__}"
        )
    if weights.w13_scale is not None or weights.w2_scale is not None:
        raise MoEEpConfigError(
            "sm90_push_bf16 preprocessing accepts canonical BF16 weights only; "
            "pass Sm90PushBf16Weights through MegaConfig.transformed_weights "
            "to reuse a prepared bundle"
        )
    expected = (
        ("w13", weights.w13, (num_local_experts, 2 * intermediate_size, hidden_size)),
        ("w2", weights.w2, (num_local_experts, hidden_size, intermediate_size)),
    )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise MoEEpConfigError(
                f"sm90_push_bf16 {name} must have shape {shape}, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"sm90_push_bf16 {name} must be torch.bfloat16, got {tensor.dtype}"
            )
        if not tensor.is_cuda:
            raise MoEEpConfigError(f"sm90_push_bf16 {name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"sm90_push_bf16 {name} must be contiguous")
    if weights.w13.device != weights.w2.device:
        raise MoEEpConfigError(
            "sm90_push_bf16 w13 and w2 must be on the same CUDA device"
        )
    transformed = make_sm90_push_bf16_weights(weights.w13, weights.w2)
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
