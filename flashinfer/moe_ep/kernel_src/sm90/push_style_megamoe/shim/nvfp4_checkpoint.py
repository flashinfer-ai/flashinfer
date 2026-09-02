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

Canonical NVFP4 checkpoint contract.

Two E2M1 values occupy each byte.  The low nibble is the even K element and
the high nibble is the following odd K element.  E4M3FN scales are linear
``[expert, output, ceil(K / 16)]`` values; each scale applies to sixteen K
elements before a positive FP32 per-tensor or per-expert ``global_alpha``
multiplier.  A scalar is stored as a zero-dimensional tensor, while a
per-expert multiplier is stored as ``[E]``.
Physical tensors may contain N/K padding, while ``logical_shape`` identifies
the visible ``[E, N, K]`` region.

Producers quantize E2M1 and E4M3FN with round-to-nearest-even and finite
saturation at 6 and 448 respectively.  Checkpoint tensors already contain
encoded values, so decoding performs no further rounding or saturation.
ModelOpt ``weight_scale_2`` is the positive global decode multiplier
``amax / (6 * 448)`` and is loaded directly as ``global_alpha``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_VALIDATION_CHUNK_ELEMENTS = 8 * 1024 * 1024


def _validate_e4m3_scales(scales: torch.Tensor) -> None:
    raw = scales.detach().reshape(-1).view(torch.uint8)
    for begin in range(0, raw.numel(), _VALIDATION_CHUNK_ELEMENTS):
        chunk = raw[begin : begin + _VALIDATION_CHUNK_ELEMENTS]
        if bool((chunk.bitwise_and(0x7F) == 0x7F).any()):
            raise ValueError("scale_e4m3_per16 must contain finite values")
        if bool((chunk == 0x80).any()):
            raise ValueError("scale_e4m3_per16 must not contain negative zero")
        if bool((chunk.bitwise_and(0x80) != 0).any()):
            raise ValueError("scale_e4m3_per16 must contain non-negative values")


def _require_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _normalize_shape(logical_shape: Sequence[int]) -> tuple[int, int, int]:
    if not isinstance(logical_shape, Sequence) or isinstance(logical_shape, str):
        raise TypeError("logical_shape must be a three-element sequence")
    if len(logical_shape) != 3:
        raise ValueError(
            f"logical_shape must contain three dimensions, got {len(logical_shape)}"
        )
    shape = (int(logical_shape[0]), int(logical_shape[1]), int(logical_shape[2]))
    if shape[0] < 0 or shape[1] <= 0 or shape[2] <= 0:
        raise ValueError(f"logical_shape must be [E>=0,N>0,K>0], got {shape}")
    return shape


def _normalize_expert_mapping(
    expert_mapping: Sequence[int], experts: int
) -> tuple[int, ...]:
    if not isinstance(expert_mapping, Sequence) or isinstance(expert_mapping, str):
        raise TypeError("expert_mapping must be a sequence")
    mapping = tuple(int(expert) for expert in expert_mapping)
    if len(mapping) != experts:
        raise ValueError(
            f"expert_mapping must contain E={experts} entries, got {len(mapping)}"
        )
    if any(expert < 0 for expert in mapping):
        raise ValueError("expert_mapping entries must be non-negative")
    if len(set(mapping)) != len(mapping):
        raise ValueError("expert_mapping entries must be unique")
    return mapping


def _normalize_alpha(
    alpha: torch.Tensor, experts: int, device: torch.device
) -> torch.Tensor:
    _require_tensor("global_alpha", alpha)
    if alpha.dtype != torch.float32:
        raise ValueError(
            f"global_alpha must have dtype torch.float32, got {alpha.dtype}"
        )
    if alpha.device != device:
        raise ValueError(f"global_alpha must be on {device}, got {alpha.device}")
    if not alpha.is_contiguous():
        raise ValueError("global_alpha must be contiguous")
    if alpha.ndim == 0:
        alpha = alpha.reshape(())
    elif (
        alpha.ndim >= 1
        and alpha.shape[0] == experts
        and all(size == 1 for size in alpha.shape[1:])
    ):
        alpha = alpha.reshape(experts)
    else:
        raise ValueError(
            f"global_alpha must be scalar or per-expert singleton-shaped for "
            f"E={experts}, got {tuple(alpha.shape)}"
        )
    if alpha.numel() and (
        not bool(torch.isfinite(alpha).all()) or bool((alpha <= 0).any())
    ):
        raise ValueError("global_alpha must contain finite positive multipliers")
    return alpha


@dataclass(frozen=True)
class NVFP4Checkpoint:
    packed_e2m1: torch.Tensor
    scale_e4m3_per16: torch.Tensor
    global_alpha: torch.Tensor
    logical_shape: tuple[int, int, int]
    expert_mapping: tuple[int, ...]
    source_format_version: str

    def __post_init__(self) -> None:
        _require_tensor("packed_e2m1", self.packed_e2m1)
        _require_tensor("scale_e4m3_per16", self.scale_e4m3_per16)
        if self.packed_e2m1.dtype != torch.uint8 or self.packed_e2m1.ndim != 3:
            raise ValueError("packed_e2m1 must be contiguous uint8 [E,N,K/2]")
        if (
            self.scale_e4m3_per16.dtype != torch.float8_e4m3fn
            or self.scale_e4m3_per16.ndim != 3
        ):
            raise ValueError("scale_e4m3_per16 must be contiguous E4M3FN [E,N,K/16]")
        if not self.packed_e2m1.is_contiguous():
            raise ValueError("packed_e2m1 must be contiguous")
        if not self.scale_e4m3_per16.is_contiguous():
            raise ValueError("scale_e4m3_per16 must be contiguous")
        if self.packed_e2m1.device != self.scale_e4m3_per16.device:
            raise ValueError("packed_e2m1 and scale_e4m3_per16 must share a device")

        logical_shape = _normalize_shape(self.logical_shape)
        experts, padded_rows, packed_k = self.packed_e2m1.shape
        padded_k = packed_k * 2
        if padded_rows <= 0 or padded_k <= 0 or padded_k % 16:
            raise ValueError(
                "packed_e2m1 physical N must be positive and physical K must be "
                "a positive multiple of 16"
            )
        if logical_shape[0] != experts:
            raise ValueError(
                f"logical_shape E must match packed_e2m1 E={experts}, "
                f"got {logical_shape[0]}"
            )
        if logical_shape[1] > padded_rows or logical_shape[2] > padded_k:
            raise ValueError(
                f"logical_shape {logical_shape} exceeds physical shape "
                f"{(experts, padded_rows, padded_k)}"
            )
        expected_scales = (experts, padded_rows, padded_k // 16)
        if tuple(self.scale_e4m3_per16.shape) != expected_scales:
            raise ValueError(
                f"scale_e4m3_per16 must have shape {expected_scales}, got "
                f"{tuple(self.scale_e4m3_per16.shape)}"
            )
        _validate_e4m3_scales(self.scale_e4m3_per16)

        alpha = _normalize_alpha(self.global_alpha, experts, self.packed_e2m1.device)
        mapping = _normalize_expert_mapping(self.expert_mapping, experts)
        if not isinstance(self.source_format_version, str):
            raise TypeError("source_format_version must be str")
        source_format_version = self.source_format_version.strip()
        if not source_format_version:
            raise ValueError("source_format_version must not be empty")
        object.__setattr__(self, "global_alpha", alpha)
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "expert_mapping", mapping)
        object.__setattr__(self, "source_format_version", source_format_version)

    @property
    def physical_shape(self) -> tuple[int, int, int]:
        experts, rows, packed_k = self.packed_e2m1.shape
        return experts, rows, packed_k * 2

    @property
    def device(self) -> torch.device:
        return self.packed_e2m1.device

    @property
    def alpha_scope(self) -> str:
        return "per_tensor" if self.global_alpha.ndim == 0 else "per_expert"

    @property
    def global_alpha_per_expert(self) -> torch.Tensor:
        if self.global_alpha.ndim == 0:
            return self.global_alpha.expand(self.logical_shape[0])
        return self.global_alpha


@torch.no_grad()
def reference_dequantize_nvfp4(checkpoint: NVFP4Checkpoint) -> torch.Tensor:
    if not isinstance(checkpoint, NVFP4Checkpoint):
        raise TypeError("checkpoint must be an NVFP4Checkpoint")
    low = checkpoint.packed_e2m1.bitwise_and(0x0F)
    high = checkpoint.packed_e2m1.bitwise_right_shift(4).bitwise_and(0x0F)
    codes = torch.stack((low, high), dim=-1).reshape(checkpoint.physical_shape)
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=checkpoint.device)
    decoded = values[codes.to(torch.int64)]
    scales = checkpoint.scale_e4m3_per16.to(torch.float32).repeat_interleave(16, dim=-1)
    decoded = decoded * scales * checkpoint.global_alpha_per_expert[:, None, None]
    _, rows, columns = checkpoint.logical_shape
    return decoded[:, :rows, :columns].contiguous()


def _modelopt_key(prefix: str, suffix: str) -> str:
    normalized = prefix.rstrip(".")
    return f"{normalized}.{suffix}" if normalized else suffix


def _state_tensor(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state_dict:
        raise KeyError(f"missing ModelOpt NVFP4 tensor {key!r}")
    value = state_dict[key]
    _require_tensor(key, value)
    return value


def load_modelopt_nvfp4_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = "",
    logical_shape: Sequence[int] | None = None,
    expert_mapping: Sequence[int] | None = None,
    source_format_version: str = "modelopt.nvfp4.v1",
) -> NVFP4Checkpoint:
    """Load the linear ModelOpt NVFP4 ``weight``/scale tensor convention.

    ``weight`` is uint8 E2M1x2 with low-nibble-first K order,
    ``weight_scale`` is linear E4M3FN ``[E,N,K/16]``, and
    ``weight_scale_2`` is ModelOpt's positive FP32 global decode multiplier.
    Two-dimensional tensors are treated as one expert.
    Swizzled runtime scale-factor layouts are rejected rather than inferred.
    """

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be str")
    packed = _state_tensor(state_dict, _modelopt_key(prefix, "weight"))
    scales = _state_tensor(state_dict, _modelopt_key(prefix, "weight_scale"))
    global_decode_scale = _state_tensor(
        state_dict, _modelopt_key(prefix, "weight_scale_2")
    )
    if packed.ndim not in (2, 3):
        raise ValueError(
            f"ModelOpt weight must be [N,K/2] or [E,N,K/2], got {tuple(packed.shape)}"
        )
    if scales.ndim not in (2, 3):
        raise ValueError(
            "ModelOpt weight_scale must be [N,K/16] or [E,N,K/16], got "
            f"{tuple(scales.shape)}"
        )
    packed_was_2d = packed.ndim == 2
    if packed.ndim == 2:
        packed = packed.unsqueeze(0)
    if scales.ndim == 2:
        scales = scales.unsqueeze(0)
    if packed.dtype != torch.uint8:
        raise ValueError(
            f"ModelOpt weight must have dtype torch.uint8, got {packed.dtype}"
        )
    if scales.dtype == torch.uint8:
        scales = scales.view(torch.float8_e4m3fn)
    if scales.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"ModelOpt weight_scale must contain E4M3FN values, got {scales.dtype}"
        )
    if global_decode_scale.dtype != torch.float32:
        raise ValueError(
            "ModelOpt weight_scale_2 must have dtype torch.float32, got "
            f"{global_decode_scale.dtype}"
        )
    if global_decode_scale.numel() and (
        not bool(torch.isfinite(global_decode_scale).all())
        or bool((global_decode_scale <= 0).any())
    ):
        raise ValueError("ModelOpt weight_scale_2 must be finite and positive")
    packed = packed.contiguous()
    scales = scales.contiguous()
    # A ModelOpt linear layer stores one tensor-wide second-level scale as [1].
    # Collapse only this source convention; NVFP4Checkpoint itself deliberately
    # preserves an E=1 vector as per-expert metadata.
    global_alpha = global_decode_scale.contiguous()
    if packed_was_2d and global_alpha.numel() == 1:
        global_alpha = global_alpha.reshape(())
    physical_shape = (packed.shape[0], packed.shape[1], packed.shape[2] * 2)
    if logical_shape is None:
        logical_shape = physical_shape
    if expert_mapping is None:
        expert_mapping = tuple(range(physical_shape[0]))
    return NVFP4Checkpoint(
        packed,
        scales,
        global_alpha,
        _normalize_shape(logical_shape),
        tuple(expert_mapping),
        source_format_version,
    )


__all__ = [
    "NVFP4Checkpoint",
    "load_modelopt_nvfp4_state_dict",
    "reference_dequantize_nvfp4",
]
