# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weights of the SM120 MXFP4 x MXFP8 nosync MegaMoE kernel.

Canonical input (same contract as the ``mxfp4_mxfp8_bf16_cake`` and
``mxfp4_mxfp8_bf16_cutedsl`` backends): a
:class:`~flashinfer.moe_ep.weights.PrequantizedMoEWeights` with

* ``w13``: packed E2M1 ``[local_E, 2 * I, H // 2]`` (gate rows then up rows,
  low nibble first), ``uint8`` / ``int8`` / ``float4_e2m1fn_x2``;
* ``w2``: packed E2M1 ``[local_E, H, I // 2]``;
* ``w13_scale``: UE8M0 exponent bytes ``[local_E, 2 * I, H // 32]``;
* ``w2_scale``: UE8M0 exponent bytes ``[local_E, H, I // 32]``.

Kernel layout: the packed E2M1 data row-major with the ``w13`` rows interleaved
per 16-row block (gate rows ``8b..8b+7``, then up rows ``8b..8b+7``), which is
exactly what the cake and cutedsl backends keep, so a layer that switches between
this kernel and one of them shares their tensors (:func:`shared_fp4`); and the
UE8M0 scales per 16-row block and K128, 16 rows x 4 bytes (one word per row), in
the same row order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ......weights import MoEWeightPack, PrequantizedMoEWeights

FP4_PACKED_DTYPES = (torch.uint8, torch.int8, torch.float4_e2m1fn_x2)
SCALE_DTYPES = (torch.uint8, torch.float8_e8m0fnu)


@dataclass(frozen=True)
class NosyncMegaWeights:
    w13: torch.Tensor  # uint8 [local_E, 2I, H/2], gate/up rows interleaved by 8
    w2: torch.Tensor  # uint8 [local_E, H, I/2]
    w13_scale: torch.Tensor  # uint8 [local_E, 2I * H / 32]
    w2_scale: torch.Tensor  # uint8 [local_E, H * I / 32]


def _w13_rows(intermediate: int, device: torch.device) -> torch.Tensor:
    """Kernel row -> checkpoint row of ``w13``: per 16-row block, gate rows 8b..8b+7 then up rows 8b..8b+7."""
    r = torch.arange(2 * intermediate, device=device)
    blk, w = r // 16, r % 16
    return torch.where(w < 8, blk * 8 + w, intermediate + blk * 8 + (w - 8))


def _scales(sf: torch.Tensor) -> torch.Tensor:
    """[E, N, K/32] UE8M0 -> per 16-row block and K128: 16 rows x 4 bytes (one word per row)."""
    e, n, ks = sf.shape
    return sf.reshape(e, n // 16, 16, ks // 4, 4).permute(0, 1, 3, 2, 4).reshape(e, -1)


def is_mxfp4_pack(weights: Any) -> bool:
    """Whether ``weights`` carries MXFP4 experts this kernel can run."""
    return (
        isinstance(weights, PrequantizedMoEWeights)
        and weights.w13.dtype in FP4_PACKED_DTYPES
        and weights.w2.dtype in FP4_PACKED_DTYPES
        and weights.w13_scale.dtype in SCALE_DTYPES
        and weights.w2_scale.dtype in SCALE_DTYPES
    )


def _checked(weights: MoEWeightPack, hidden: int, intermediate: int):
    if not is_mxfp4_pack(weights):
        raise NotImplementedError(
            "SM120 MXFP4 x MXFP8 requires pre-quantized packed E2M1 weights "
            "and both UE8M0 scale planes"
        )
    w13, w2 = weights.w13.view(torch.uint8), weights.w2.view(torch.uint8)
    s13, s2 = weights.w13_scale.view(torch.uint8), weights.w2_scale.view(torch.uint8)
    e, h, i = w13.shape[0], hidden, intermediate
    for name, tensor, shape in (
        ("w13", w13, (e, 2 * i, h // 2)),
        ("w2", w2, (e, h, i // 2)),
        ("w13_scale", s13, (e, 2 * i, h // 32)),
        ("w2_scale", s2, (e, h, i // 32)),
    ):
        if tuple(tensor.shape) != shape or not tensor.is_cuda:
            raise ValueError(
                f"{name} must be a CUDA tensor of shape {shape}, got {tuple(tensor.shape)}"
            )
    return w13, w2, s13, s2


def shared_fp4(
    transformed: Any, kernel_name: str | None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The packed E2M1 ``w13`` / ``w2`` inside the transformed weights of the backend named
    ``kernel_name`` when that backend keeps them in this kernel's layout (cake, cutedsl),
    else ``None``."""
    if kernel_name == "sm120_mxfp4_mxfp8_bf16_cake":
        w13 = getattr(transformed, "w1_fp4", None)
        w2 = getattr(transformed, "w2_fp4", None)
    elif kernel_name == "sm120_mxfp4_mxfp8_bf16_cutedsl":
        try:  # ((fc1, fc1_scale), (fc2, fc2_scale))
            (w13, _), (w2, _) = transformed
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not all(
        isinstance(t, torch.Tensor) and t.dtype in FP4_PACKED_DTYPES and t.ndim == 3
        for t in (w13, w2)
    ):
        return None
    return w13.view(torch.uint8), w2.view(torch.uint8)


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    hidden_size: int,
    intermediate_size: int,
    fp4: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> NosyncMegaWeights:
    """The kernel's weights: ``fp4`` (another backend's tensors in this layout, see
    :func:`shared_fp4`) or a copy of the packed data, plus the scales."""
    w13, w2, s13, s2 = _checked(weights, hidden_size, intermediate_size)
    rows = _w13_rows(intermediate_size, w13.device)
    if fp4 is None:
        fp4 = (w13[:, rows].contiguous(), w2.contiguous())
    return NosyncMegaWeights(
        w13=fp4[0],
        w2=fp4[1],
        w13_scale=_scales(s13[:, rows]).contiguous(),
        w2_scale=_scales(s2).contiguous(),
    )


def validate_transformed_mega_weights(
    transformed: NosyncMegaWeights,
    *,
    hidden_size: int,
    intermediate_size: int,
    local_experts: int,
) -> None:
    if not isinstance(transformed, NosyncMegaWeights):
        raise ValueError("transformed weights must be NosyncMegaWeights")
    h, i, e = hidden_size, intermediate_size, local_experts
    for name, tensor, shape in (
        ("w13", transformed.w13, (e, 2 * i, h // 2)),
        ("w2", transformed.w2, (e, h, i // 2)),
        ("w13_scale", transformed.w13_scale, (e, 2 * i * h // 32)),
        ("w2_scale", transformed.w2_scale, (e, h * i // 32)),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.uint8
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous CUDA uint8 tensor of shape {shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )


__all__ = [
    "NosyncMegaWeights",
    "is_mxfp4_pack",
    "preprocess_mega_weights",
    "shared_fp4",
    "validate_transformed_mega_weights",
]
