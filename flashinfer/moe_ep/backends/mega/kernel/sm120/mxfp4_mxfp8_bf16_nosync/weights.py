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

Set ``interleave_w13=False`` to borrow the canonical gate/up-contiguous FP4 tensor
without a packed-data copy. ``scale_layout="canonical"`` also borrows both canonical
per-row scale planes. ``scale_layout="deepgemm"`` instead accepts packed UE8M0
``int32`` scales with shapes ``[local_E, 2I, H/128]`` and ``[local_E, H, I/128]``,
stored MN-major with four consecutive K32 exponents per word, and reads them in
This is the ordinary grouped-GEMM scale layout, not DeepGEMM MegaMoE's
additional UTCCP-swizzled layout. Layout mismatches are rejected instead of
silently computing with wrong scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from ......weights import MoEWeightPack, PrequantizedMoEWeights
from ..mxfp4_mxfp8_bf16_cake.cake_weights import (
    FP4_PACKED_DTYPES,
    SCALE_DTYPES,
    gate_up_interleave_index,
)


@dataclass(frozen=True)
class NosyncMegaWeights:
    w13: torch.Tensor  # uint8 [local_E, 2I, H/2]
    w2: torch.Tensor  # uint8 [local_E, H, I/2]
    w13_scale: torch.Tensor  # uint8 [local_E, 2I * H / 32]
    w2_scale: torch.Tensor  # uint8 [local_E, H * I / 32]
    w13_gate_up_interleaved: bool = True
    scale_layout: Literal["kernel", "canonical", "deepgemm"] = "kernel"


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


def _deepgemm_mn_major(sf: torch.Tensor) -> torch.Tensor:
    if sf.dtype != torch.int32 or sf.ndim != 3:
        raise ValueError("DeepGEMM scales must be rank-3 packed int32 tensors")
    e, n, k = sf.shape
    aligned_n = (n + 3) // 4 * 4
    if n % 4 or sf.stride() != (k * aligned_n, 1, aligned_n):
        raise ValueError(
            "DeepGEMM scales must be MN-major int32 with four packed UE8M0 "
            f"exponents per K128 word, got shape {tuple(sf.shape)} "
            f"stride {sf.stride()}"
        )
    return torch.empty(0, dtype=torch.uint8, device=sf.device).set_(
        sf.untyped_storage(),
        sf.storage_offset() * 4,
        (e, k, n * 4),
        (sf.stride(0) * 4, sf.stride(2) * 4, 1),
    )


def _checked(weights: MoEWeightPack, hidden: int, intermediate: int, scale_layout: str):
    if scale_layout not in ("kernel", "canonical", "deepgemm"):
        raise ValueError(f"unsupported scale layout {scale_layout!r}")
    if scale_layout == "deepgemm":
        if (
            not isinstance(weights, PrequantizedMoEWeights)
            or weights.w13.dtype not in FP4_PACKED_DTYPES
            or weights.w2.dtype not in FP4_PACKED_DTYPES
            or weights.w13_scale.dtype not in (*SCALE_DTYPES, torch.int32)
            or weights.w2_scale.dtype not in (*SCALE_DTYPES, torch.int32)
        ):
            raise NotImplementedError(
                "DeepGEMM scales require packed E2M1 weights and UE8M0 scale planes"
            )
        if hidden % 128 or intermediate % 128:
            raise ValueError("DeepGEMM scales require K aligned to 128")
        s13, s2 = weights.w13_scale, weights.w2_scale
        if weights.w13_scale.dtype == torch.int32:
            s13 = _deepgemm_mn_major(s13)
        if weights.w2_scale.dtype == torch.int32:
            s2 = _deepgemm_mn_major(s2)
    else:
        if not is_mxfp4_pack(weights):
            raise NotImplementedError(
                "SM120 MXFP4 x MXFP8 requires pre-quantized packed E2M1 weights "
                "and both UE8M0 scale planes"
            )
        s13, s2 = weights.w13_scale, weights.w2_scale
    w13, w2 = weights.w13.view(torch.uint8), weights.w2.view(torch.uint8)
    s13, s2 = s13.view(torch.uint8), s2.view(torch.uint8)
    e, h, i = w13.shape[0], hidden, intermediate
    for name, tensor, shape in (
        ("w13", w13, (e, 2 * i, h // 2)),
        ("w2", w2, (e, h, i // 2)),
        (
            "w13_scale",
            s13,
            (e, h // 128, 8 * i) if scale_layout == "deepgemm" else (e, 2 * i, h // 32),
        ),
        (
            "w2_scale",
            s2,
            (e, i // 128, 4 * h) if scale_layout == "deepgemm" else (e, h, i // 32),
        ),
    ):
        if tuple(tensor.shape) != shape or not tensor.is_cuda:
            raise ValueError(
                f"{name} must be a CUDA tensor of shape {shape}, "
                f"got {tuple(tensor.shape)}"
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
    interleave_w13: bool = True,
    scale_layout: Literal["kernel", "canonical", "deepgemm"] = "kernel",
) -> NosyncMegaWeights:
    """Prepare weights and optionally borrow the FP4 and scale tensors.

    ``deepgemm`` takes DeepGEMM's MN-major packed UE8M0 scales directly. The
    returned tensors share the source allocation and its gate/up row order.
    """
    if not interleave_w13 and scale_layout == "kernel":
        raise ValueError("gate/up-contiguous FP4 requires canonical or DeepGEMM scales")
    w13, w2, s13, s2 = _checked(weights, hidden_size, intermediate_size, scale_layout)
    rows = gate_up_interleave_index(intermediate_size).to(w13.device)
    if fp4 is None:
        fp4 = (
            (w13[:, rows].contiguous(), w2.contiguous())
            if interleave_w13
            else (w13, w2)
        )
    return NosyncMegaWeights(
        w13=fp4[0],
        w2=fp4[1],
        w13_scale=(
            _scales(s13[:, rows]).contiguous() if scale_layout == "kernel" else s13
        ),
        w2_scale=(_scales(s2).contiguous() if scale_layout == "kernel" else s2),
        w13_gate_up_interleaved=interleave_w13,
        scale_layout=scale_layout,
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
    if transformed.scale_layout not in ("kernel", "canonical", "deepgemm"):
        raise ValueError(f"unsupported scale layout {transformed.scale_layout!r}")
    if not transformed.w13_gate_up_interleaved and transformed.scale_layout == "kernel":
        raise ValueError("gate/up-contiguous FP4 requires canonical or DeepGEMM scales")
    h, i, e = hidden_size, intermediate_size, local_experts
    scale_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    if transformed.scale_layout == "deepgemm":
        scale_shapes = ((e, h // 128, 8 * i), (e, i // 128, 4 * h))
        scale_strides = (
            (2 * i * h // 32, 8 * i, 1),
            (h * i // 32, 4 * h, 1),
        )
    elif transformed.scale_layout == "canonical":
        scale_shapes = ((e, 2 * i, h // 32), (e, h, i // 32))
        scale_strides = None
    else:
        scale_shapes = ((e, 2 * i * h // 32), (e, h * i // 32))
        scale_strides = None
    for name, tensor, shape, strides in (
        ("w13", transformed.w13, (e, 2 * i, h // 2), None),
        ("w2", transformed.w2, (e, h, i // 2), None),
        ("w13_scale", transformed.w13_scale, scale_shapes[0], scale_strides),
        ("w2_scale", transformed.w2_scale, scale_shapes[1], scale_strides),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.uint8
            or not tensor.is_cuda
        ):
            raise ValueError(
                f"{name} must be a CUDA uint8 tensor of shape {shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
        if strides is None and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got {tensor.stride()}")
        if strides is not None and tensor.stride() != tuple(strides):
            raise ValueError(
                f"{name} must have stride {strides}, got {tensor.stride()}"
            )


__all__ = [
    "NosyncMegaWeights",
    "is_mxfp4_pack",
    "preprocess_mega_weights",
    "shared_fp4",
    "validate_transformed_mega_weights",
]
