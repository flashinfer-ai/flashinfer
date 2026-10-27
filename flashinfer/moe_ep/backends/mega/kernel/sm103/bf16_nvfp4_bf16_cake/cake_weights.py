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
"""

# Expert weights of the Cake W4A16 MegaMoE route: the canonical
# ``PrequantizedMoEWeights`` layout, consumed as is.
#
# The programs read the canonical NVFP4 pack directly: ``w13`` uint8
# ``[E_local, 2 * I, H / 2]`` (two E2M1 codes per byte, low nibble first, gate
# rows then up rows), ``w13_scale`` float8_e4m3fn ``[E_local, 2 * I, H / 16]``,
# ``w2`` uint8 ``[E_local, H, I / 2]`` and ``w2_scale`` ``[E_local, H, I / 16]``.
# There is no prepare step, no interleave and no BF16 expansion: the gate/up
# remap and the E2M1 x E4M3 decode happen in the transform warps. The same
# pack therefore serves this route and the routed grouped-GEMM route.

from __future__ import annotations

from dataclasses import dataclass

import torch

from flashinfer.moe_ep.weights import MoEWeightPack, PrequantizedMoEWeights

from .cake_plan import FP4_PER_BYTE, KERNEL_NAME, SCALE_BLOCK


@dataclass(frozen=True)
class TransformedCakeW4a16MegaWeights:
    """The canonical NVFP4 pack of this rank's experts, validated and contiguous."""

    w13: torch.Tensor  # uint8 [E_local, 2 * I, H / 2]
    w13_scale: torch.Tensor  # float8_e4m3fn [E_local, 2 * I, H / 16]
    w2: torch.Tensor  # uint8 [E_local, H, I / 2]
    w2_scale: torch.Tensor  # float8_e4m3fn [E_local, H, I / 16]

    @property
    def local_experts(self) -> int:
        return int(self.w13.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.w13.shape[2]) * FP4_PER_BYTE

    @property
    def intermediate_size(self) -> int:
        return int(self.w13.shape[1]) // 2

    @property
    def device(self) -> torch.device:
        return self.w13.device


def _as_uint8_codes(weight: torch.Tensor, name: str) -> torch.Tensor:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype == torch.uint8:
        return weight
    if fp4_dtype is not None and weight.dtype == fp4_dtype:
        return weight.view(torch.uint8)
    raise ValueError(
        f"{KERNEL_NAME}: {name} must hold packed E2M1 codes as uint8 (or "
        f"float4_e2m1fn_x2), got {weight.dtype}"
    )


def _check(
    weight: torch.Tensor,
    scale: torch.Tensor,
    name: str,
    *,
    experts: int,
    rows: int,
    columns: int,
) -> None:
    if tuple(weight.shape) != (experts, rows, columns // FP4_PER_BYTE):
        raise ValueError(
            f"{KERNEL_NAME}: {name} must have shape "
            f"{(experts, rows, columns // FP4_PER_BYTE)} (packed E2M1), got "
            f"{tuple(weight.shape)}"
        )
    if scale.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"{KERNEL_NAME}: {name}_scale must be float8_e4m3fn, got {scale.dtype}"
        )
    if tuple(scale.shape) != (experts, rows, columns // SCALE_BLOCK):
        raise ValueError(
            f"{KERNEL_NAME}: {name}_scale must have shape "
            f"{(experts, rows, columns // SCALE_BLOCK)} (one E4M3 per 16 codes), got "
            f"{tuple(scale.shape)}"
        )
    if scale.device != weight.device:
        raise ValueError(
            f"{KERNEL_NAME}: {name} and {name}_scale must be on the same device"
        )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedCakeW4a16MegaWeights:
    """Validate the canonical NVFP4 pack of this rank's experts.

    No layout change and no decode: the result holds the (contiguous) input
    tensors. A BF16 / FP32 pack is refused; quantize offline to the canonical
    ``PrequantizedMoEWeights`` layout.
    """

    if not isinstance(weights, PrequantizedMoEWeights):
        raise ValueError(
            f"{KERNEL_NAME} consumes PrequantizedMoEWeights (packed E2M1 codes with "
            f"E4M3 block-16 scales); got {type(weights).__name__}. Quantize the "
            "expert weights offline."
        )
    if hidden_size % (FP4_PER_BYTE * SCALE_BLOCK) or intermediate_size % (
        FP4_PER_BYTE * SCALE_BLOCK
    ):
        raise ValueError(
            f"{KERNEL_NAME}: hidden_size and intermediate_size must be multiples of "
            f"{FP4_PER_BYTE * SCALE_BLOCK}, got {hidden_size} and {intermediate_size}"
        )
    w13 = _as_uint8_codes(weights.w13, "w13")
    w2 = _as_uint8_codes(weights.w2, "w2")
    experts = int(w13.shape[0]) if w13.ndim == 3 else -1
    if w13.ndim != 3 or w2.ndim != 3 or experts <= 0:
        raise ValueError(f"{KERNEL_NAME}: w13 and w2 must be 3-D [E_local, rows, K/2]")
    _check(
        w13,
        weights.w13_scale,
        "w13",
        experts=experts,
        rows=2 * intermediate_size,
        columns=hidden_size,
    )
    _check(
        w2,
        weights.w2_scale,
        "w2",
        experts=experts,
        rows=hidden_size,
        columns=intermediate_size,
    )
    if w2.device != w13.device:
        raise ValueError(f"{KERNEL_NAME}: w13 and w2 must be on the same device")
    return TransformedCakeW4a16MegaWeights(
        w13=w13.contiguous(),
        w13_scale=weights.w13_scale.contiguous(),
        w2=w2.contiguous(),
        w2_scale=weights.w2_scale.contiguous(),
    )


def validate_transformed_mega_weights(
    transformed: object,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
    device: torch.device | None = None,
) -> TransformedCakeW4a16MegaWeights:
    """Check a caller-supplied :class:`TransformedCakeW4a16MegaWeights`."""

    if not isinstance(transformed, TransformedCakeW4a16MegaWeights):
        raise ValueError(
            f"{KERNEL_NAME}: transformed weights must be "
            f"TransformedCakeW4a16MegaWeights, got {type(transformed).__name__}"
        )
    local_experts = num_experts // world_size
    if transformed.local_experts != local_experts:
        raise ValueError(
            f"{KERNEL_NAME}: transformed weights hold {transformed.local_experts} "
            f"experts; this rank owns {local_experts}"
        )
    for name, tensor in (
        ("w13", transformed.w13),
        ("w13_scale", transformed.w13_scale),
        ("w2", transformed.w2),
        ("w2_scale", transformed.w2_scale),
    ):
        if device is None and not tensor.is_cuda:
            raise ValueError(f"{KERNEL_NAME}: {name} must be a CUDA tensor")
        if device is not None and tensor.device != device:
            raise ValueError(
                f"{KERNEL_NAME}: {name} is on {tensor.device}, the workspace on {device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{KERNEL_NAME}: {name} must be contiguous")
    if transformed.w13.dtype != torch.uint8 or transformed.w2.dtype != torch.uint8:
        raise ValueError(f"{KERNEL_NAME}: packed weights must be uint8")
    _check(
        transformed.w13,
        transformed.w13_scale,
        "w13",
        experts=local_experts,
        rows=2 * intermediate_size,
        columns=hidden_size,
    )
    _check(
        transformed.w2,
        transformed.w2_scale,
        "w2",
        experts=local_experts,
        rows=hidden_size,
        columns=intermediate_size,
    )
    return transformed


__all__ = [
    "TransformedCakeW4a16MegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
