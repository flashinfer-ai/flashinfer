"""Canonical-to-BlockMajorK weights for the rank-major BF16 backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ......core.validation.common import MoEEpConfigError
from ......weights import MoEWeightPack

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TransformedMegaWeights:
    """Kernel-ready BF16 weights in physical BlockMajorK layout."""

    w13_block_major: "torch.Tensor"
    w2_block_major: "torch.Tensor"


def _shuffle_rows(weight: "torch.Tensor", *, gated: bool) -> "torch.Tensor":
    """Apply the TRT-LLM 32-row permutation used by the generated kernels."""
    import torch

    experts, rows, columns = map(int, weight.shape)
    if gated:
        half = rows // 2
        # The generated FC1 consumes linear/up first and gate second, then
        # interleaves corresponding rows before applying the 32-row shuffle.
        weight = torch.stack((weight[:, :half], weight[:, half:]), dim=2).reshape(
            experts,
            rows,
            columns,
        )
    logical_row = torch.arange(rows, device=weight.device)
    row_in_block = logical_row % 32
    physical_row = (logical_row // 32) * 32 + (row_in_block % 4) * 8 + row_in_block // 4
    inverse = torch.empty_like(physical_row)
    inverse[physical_row] = logical_row
    return weight.index_select(1, inverse)


def _block_major(weight: "torch.Tensor") -> "torch.Tensor":
    experts, rows, columns = map(int, weight.shape)
    return (
        weight.reshape(experts, rows, columns // 64, 64)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
) -> TransformedMegaWeights:
    """Convert gate-first/up-second canonical BF16 weights to BlockMajorK.

    ``MoEWeightPack.w13`` follows FlashInfer's canonical ``[gate, up]`` row
    order.  The generated kernel's logical FC1 order is ``[up, gate]``.  The
    half swap below is therefore semantic, and precedes the kernel's own
    gate-pair interleave and TRT-LLM row permutation.
    """
    import torch

    if not isinstance(weights, MoEWeightPack):
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda weights must be "
            f"MoEWeightPack, got {type(weights).__name__}"
        )
    if weights.w13_scale is not None or weights.w2_scale is not None:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda accepts canonical BF16 "
            "weights without scale planes"
        )

    expected_w13 = (num_local_experts, 2 * intermediate_size, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    for name, tensor, shape in (
        ("w13", weights.w13, expected_w13),
        ("w2", weights.w2, expected_w2),
    ):
        if tuple(tensor.shape) != shape:
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must have "
                f"shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must be "
                f"torch.bfloat16, got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must be contiguous"
            )
    if weights.w13.device != weights.w2.device:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda w13 and w2 must share a device"
        )
    if hidden_size % 64 or intermediate_size % 64:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda BlockMajorK dimensions "
            "must be divisible by 64"
        )

    gate, up = weights.w13.split(intermediate_size, dim=1)
    kernel_w13 = torch.cat((up, gate), dim=1)
    return TransformedMegaWeights(
        w13_block_major=_block_major(_shuffle_rows(kernel_w13, gated=True)),
        w2_block_major=_block_major(_shuffle_rows(weights.w2, gated=False)),
    )


def validate_transformed_mega_weights(
    transformed: object,
    *,
    intermediate_size: int,
    hidden_size: int,
    num_local_experts: int,
) -> None:
    """Validate the exact physical tensor ABI consumed by the CUDA session."""
    import torch

    if not isinstance(transformed, TransformedMegaWeights):
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda transformed weights must be "
            f"TransformedMegaWeights, got {type(transformed).__name__}"
        )
    expected = (
        (
            "w13_block_major",
            transformed.w13_block_major,
            (num_local_experts, hidden_size // 64, 2 * intermediate_size, 64),
        ),
        (
            "w2_block_major",
            transformed.w2_block_major,
            (num_local_experts, intermediate_size // 64, hidden_size, 64),
        ),
    )
    device = transformed.w13_block_major.device
    if not transformed.w13_block_major.is_cuda:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda transformed weights must "
            "be CUDA tensors"
        )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must have "
                f"shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must be "
                f"torch.bfloat16, got {tensor.dtype}"
            )
        if tensor.device != device:
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must be on "
                f"{device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must be contiguous"
            )


__all__ = [
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
