"""Shared MXFP8 MegaMoE weight-preprocessing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import torch


def mxfp8_data_dtype(kind: str) -> "torch.dtype":
    import torch

    return {
        "mxfp8_e4m3": torch.float8_e4m3fn,
        "mxfp8_e5m2": torch.float8_e5m2,
        "bf16_mxfp8_e4m3": torch.float8_e4m3fn,
        "bf16_mxfp8_e5m2": torch.float8_e5m2,
    }[kind]


def swizzle_expert_scales(raw_sf: "torch.Tensor") -> "torch.Tensor":
    from ......kernel_src.cutedsl_megamoe import to_blocked

    return to_blocked(raw_sf)


def interleave_gate_up(
    tensor: "torch.Tensor",
    *,
    intermediate_size: int,
    block_size: int,
    kernel_name: str,
) -> "torch.Tensor":
    if intermediate_size % (2 * block_size) != 0:
        raise ValueError(
            f"{kernel_name} requires full FC1 width to be divisible by "
            f"{2 * block_size}, got {intermediate_size}."
        )
    if tensor.shape[1] != intermediate_size:
        raise ValueError(
            f"expected FC1 tensor with shape (experts, {intermediate_size}, ...), "
            f"got {tuple(tensor.shape)}"
        )

    half = intermediate_size // 2
    gate = tensor[:, :half, :].contiguous()
    up = tensor[:, half:, :].contiguous()
    num_pairs = half // block_size
    out = tensor.new_empty(tensor.shape)
    out_view = out.view(tensor.shape[0], num_pairs, 2, block_size, tensor.shape[2])
    gate_view = gate.view(tensor.shape[0], num_pairs, block_size, tensor.shape[2])
    up_view = up.view(tensor.shape[0], num_pairs, block_size, tensor.shape[2])
    out_view[:, :, 0].copy_(gate_view)
    out_view[:, :, 1].copy_(up_view)
    return out.contiguous()


def quantize_mxfp8_weight_k_major(
    weight_k_major: "torch.Tensor", *, kind: str
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    from ......kernel_src.cutedsl_megamoe import mxfp8_quantize_per_block_32

    return mxfp8_quantize_per_block_32(
        weight_k_major.to(torch.float32), mxfp8_data_dtype(kind)
    )


def as_mxfp8_scale(scale: "torch.Tensor") -> "torch.Tensor":
    import torch

    from ......kernel_src.cutedsl_megamoe import Mxfp8ScaleDtype

    if scale.dtype == Mxfp8ScaleDtype:
        return scale
    if scale.dtype == torch.uint8:
        return scale.view(Mxfp8ScaleDtype)
    raise ValueError(
        f"MXFP8 weight scales must have dtype {Mxfp8ScaleDtype} or torch.uint8, "
        f"got {scale.dtype}"
    )


def mxfp8_swizzled_flat_sf_size(rows: int, cols: int) -> int:
    import torch

    from ......kernel_src.cutedsl_megamoe import Mxfp8ScaleDtype, to_blocked

    return to_blocked(torch.zeros(rows, cols, dtype=Mxfp8ScaleDtype)).numel()
