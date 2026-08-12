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

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .nvfp4_checkpoint import NVFP4Checkpoint


NVFP4_RS_LAYOUT_VERSION = 2
NVFP4_RS_TILE_N = 64
NVFP4_RS_TILE_K = 16
NVFP4_RS_THREADS = 128
NVFP4_RS_BYTES_PER_THREAD = 4

NVFP4_SM90_LAYOUT_VERSION = 3
NVFP4_SM90_TILE_N = 64
NVFP4_SM90_TILE_K = 32
NVFP4_SM90_K_ALIGNMENT = 128
NVFP4_SM90_GROUP_SIZES = (32, 64, 128)
NVFP4_SM90_RESIDUAL_SCHEMES = ("generic", "pow2")
NVFP4_SM90_NIBBLE_ORDER = "low_even_high_odd"
NVFP4_SM90_BYTE_ORDER = "little"
NVFP4_SM90_GLOBAL_LAYOUT = "kmajor_k32_n64_contiguous"
NVFP4_SM90_W13_LAYOUT = "gate_then_up"
NVFP4_SM90_ALPHA_SCOPES = ("per_tensor", "per_expert")
NVFP4_SM90_ROUNDING_MODE = "rne"
NVFP4_SM90_E4M3_MAX = 448.0
NVFP4_SM90_GENERIC_MARGIN = 1.0 + 2.0**-7
NVFP4_SM90_POW2_ZERO_SENTINEL = -128
_NVFP4_SM90_PROMOTION_CHUNK_ELEMENTS = 8 * 1024 * 1024
_NVFP4_SM90_HASH_CHUNK_BYTES = 16 * 1024 * 1024
_NVFP4_SM90_VALIDATION_CHUNK_ELEMENTS = 8 * 1024 * 1024
_NVFP4_SM90_CHECKSUM_DTYPE_TAGS = {
    torch.uint8: b"u8",
    torch.int8: b"i8",
    torch.bfloat16: b"bf16",
    torch.float32: b"f32",
    torch.float8_e4m3fn: b"e4m3fn",
}


def _validate_linear(payload: torch.Tensor, scales: torch.Tensor) -> None:
    if payload.dtype != torch.uint8 or payload.ndim != 3:
        raise ValueError("payload must be contiguous uint8 [E,N,K/2]")
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 3:
        raise ValueError("scales must be contiguous E4M3 [E,N,K/16]")
    if not payload.is_contiguous() or not scales.is_contiguous():
        raise ValueError("payload and scales must be contiguous")
    if payload.device != scales.device:
        raise ValueError("payload and scales must share a device")
    experts, rows, packed_k = payload.shape
    logical_k = packed_k * 2
    if rows % NVFP4_RS_TILE_N or logical_k % NVFP4_RS_TILE_K:
        raise ValueError("N and K violate NVFP4 RS tile alignment")
    if tuple(scales.shape) != (
        experts,
        rows,
        logical_k // NVFP4_RS_TILE_K,
    ):
        raise ValueError("scale shape does not match payload")


def repack_nvfp4_payload_v2(payload: torch.Tensor) -> torch.Tensor:
    if payload.dtype != torch.uint8 or payload.ndim != 3:
        raise ValueError("payload must be contiguous uint8 [E,N,K/2]")
    if not payload.is_contiguous():
        raise ValueError("payload must be contiguous")
    experts, rows, packed_k = payload.shape
    logical_k = packed_k * 2
    if rows % NVFP4_RS_TILE_N or logical_k % NVFP4_RS_TILE_K:
        raise ValueError("N and K violate NVFP4 RS tile alignment")
    n_tiles = rows // NVFP4_RS_TILE_N
    k_tiles = logical_k // NVFP4_RS_TILE_K
    return (
        payload.view(
            experts,
            n_tiles,
            4,
            2,
            8,
            k_tiles,
            2,
            4,
        )
        .permute(0, 1, 5, 2, 4, 7, 6, 3)
        .contiguous()
        .view(
            experts,
            n_tiles,
            k_tiles,
            NVFP4_RS_THREADS,
            NVFP4_RS_BYTES_PER_THREAD,
        )
    )


def unpack_nvfp4_payload_v2(payload: torch.Tensor) -> torch.Tensor:
    if payload.dtype != torch.uint8 or payload.ndim != 5:
        raise ValueError("payload must be uint8 [E,Nt,Kt,128,4]")
    if tuple(payload.shape[-2:]) != (
        NVFP4_RS_THREADS,
        NVFP4_RS_BYTES_PER_THREAD,
    ):
        raise ValueError("payload fragment shape is invalid")
    experts, n_tiles, k_tiles = payload.shape[:3]
    return (
        payload.view(
            experts,
            n_tiles,
            k_tiles,
            4,
            8,
            4,
            2,
            2,
        )
        .permute(0, 1, 3, 7, 4, 2, 6, 5)
        .contiguous()
        .view(
            experts,
            n_tiles * NVFP4_RS_TILE_N,
            k_tiles * NVFP4_RS_TILE_K // 2,
        )
    )


def repack_nvfp4_scales_v2(scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 3:
        raise ValueError("scales must be contiguous E4M3 [E,N,K/16]")
    if not scales.is_contiguous():
        raise ValueError("scales must be contiguous")
    experts, rows, k_tiles = scales.shape
    if rows % NVFP4_RS_TILE_N:
        raise ValueError("scale rows must be divisible by 64")
    return (
        scales.view(
            experts,
            rows // NVFP4_RS_TILE_N,
            NVFP4_RS_TILE_N,
            k_tiles,
        )
        .permute(0, 1, 3, 2)
        .contiguous()
    )


def unpack_nvfp4_scales_v2(scales: torch.Tensor) -> torch.Tensor:
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 4:
        raise ValueError("scales must be E4M3 [E,Nt,Kt,64]")
    if scales.shape[-1] != NVFP4_RS_TILE_N:
        raise ValueError("scale tile width must be 64")
    experts, n_tiles, k_tiles, _ = scales.shape
    return (
        scales.permute(0, 1, 3, 2)
        .contiguous()
        .view(experts, n_tiles * NVFP4_RS_TILE_N, k_tiles)
    )


@dataclass(frozen=True)
class NVFP4RSWeightView:
    payload: torch.Tensor
    scales: torch.Tensor
    alpha: torch.Tensor

    def __post_init__(self) -> None:
        if self.payload.dtype != torch.uint8 or self.payload.ndim != 5:
            raise ValueError("RS payload must be uint8")
        if self.scales.dtype != torch.float8_e4m3fn or self.scales.ndim != 4:
            raise ValueError("RS scales must be E4M3")
        if self.alpha.dtype != torch.float32 or self.alpha.ndim != 1:
            raise ValueError("RS alpha must be float32 [E]")
        if not all(
            tensor.is_contiguous() for tensor in (self.payload, self.scales, self.alpha)
        ):
            raise ValueError("RS tensors must be contiguous")
        if (
            len({tensor.device for tensor in (self.payload, self.scales, self.alpha)})
            != 1
        ):
            raise ValueError("RS tensors must share a device")
        if tuple(self.payload.shape[-2:]) != (
            NVFP4_RS_THREADS,
            NVFP4_RS_BYTES_PER_THREAD,
        ):
            raise ValueError("RS payload fragment shape is invalid")
        if self.scales.shape[-1] != NVFP4_RS_TILE_N:
            raise ValueError("RS scale tile width must be 64")
        if tuple(self.payload.shape[:3]) != tuple(self.scales.shape[:3]):
            raise ValueError("RS payload and scale tiles differ")
        if self.alpha.shape[0] != self.payload.shape[0]:
            raise ValueError("RS alpha count differs from experts")
        if not bool(torch.isfinite(self.alpha).all()) or bool((self.alpha <= 0).any()):
            raise ValueError("RS alpha must be finite and positive")


def build_nvfp4_rs_weight_view(
    payload: torch.Tensor,
    scales: torch.Tensor,
    alpha: torch.Tensor,
) -> NVFP4RSWeightView:
    _validate_linear(payload, scales)
    experts = payload.shape[0]
    if alpha.dtype != torch.float32 or alpha.device != payload.device:
        raise ValueError("alpha must be float32 on the payload device")
    if alpha.numel() == 1:
        alpha = alpha.reshape(1).expand(experts).contiguous()
    elif tuple(alpha.shape) != (experts,):
        raise ValueError("alpha must be scalar or per-expert")
    return NVFP4RSWeightView(
        repack_nvfp4_payload_v2(payload),
        repack_nvfp4_scales_v2(scales),
        alpha,
    )


def _require_exact_keys(
    name: str, value: Mapping[str, Any], expected: set[str]
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields differ: missing={missing}, extra={extra}")


def _canonical_metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _tensor_sha256(
    tensor: torch.Tensor, *, tensor_name: str, metadata: Mapping[str, Any]
) -> str:
    """Hash the stable v3 tensor preimage.

    The preimage is the domain tag, canonical JSON metadata, tensor name,
    explicit dtype tag, comma-separated decimal shape, and C-contiguous raw
    storage. Fields are NUL-delimited and multi-byte storage is little-endian.
    """

    if sys.byteorder != NVFP4_SM90_BYTE_ORDER:
        raise RuntimeError(
            "NVFP4 v3 checksums support only little-endian hosts; canonical "
            "byte swapping is not implemented"
        )
    if metadata.get("byte_order") != NVFP4_SM90_BYTE_ORDER:
        raise ValueError(
            f"NVFP4 v3 checksum metadata byte_order must be {NVFP4_SM90_BYTE_ORDER!r}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"NVFP4 v3 checksum tensor {tensor_name!r} must be contiguous")
    dtype_tag = _NVFP4_SM90_CHECKSUM_DTYPE_TAGS.get(tensor.dtype)
    if dtype_tag is None:
        raise TypeError(f"unsupported NVFP4 v3 checksum dtype {tensor.dtype}")
    shape = ",".join(str(size) for size in tensor.shape).encode("ascii")
    digest = hashlib.sha256(
        b"flashinfer.nvfp4.v3\0"
        + _canonical_metadata_bytes(metadata)
        + b"\0"
        + tensor_name.encode("ascii")
        + b"\0"
        + dtype_tag
        + b"\0"
        + shape
        + b"\0"
    )
    byte_view = tensor.detach().reshape(-1).view(torch.uint8)
    for begin in range(0, byte_view.numel(), _NVFP4_SM90_HASH_CHUNK_BYTES):
        host_chunk = byte_view[begin : begin + _NVFP4_SM90_HASH_CHUNK_BYTES].cpu()
        digest.update(memoryview(host_chunk.numpy()))
    return digest.hexdigest()


@dataclass(frozen=True)
class NVFP4V3Checksums:
    payload_sha256: str
    scale_sha256: str
    group_scale_sha256: str
    residual_sha256: str
    alpha_sha256: str

    def __post_init__(self) -> None:
        for name, digest in self.to_dict().items():
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{name} must be a 64-character SHA256 digest")
            if any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{name} must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, str]:
        return {
            "payload_sha256": self.payload_sha256,
            "scale_sha256": self.scale_sha256,
            "group_scale_sha256": self.group_scale_sha256,
            "residual_sha256": self.residual_sha256,
            "alpha_sha256": self.alpha_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NVFP4V3Checksums":
        if not isinstance(value, Mapping):
            raise TypeError("checksums must be a mapping")
        expected = {
            "payload_sha256",
            "scale_sha256",
            "group_scale_sha256",
            "residual_sha256",
            "alpha_sha256",
        }
        _require_exact_keys("checksums", value, expected)
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True)
class NVFP4V3Manifest:
    """Serialized layout-v3 contract.

    The exact field set, tensor byte layout, and each field's semantics belong
    to ``layout_version``. After version 3 is published, changing any of them
    requires layout version 4 instead of reinterpreting existing manifests.
    ``w13_layout`` names the row-concatenation order consumed by the gated
    activation: all gate rows precede all up rows.
    ``padded_shape`` uses minimal N64 and K128 padding independently of the
    selected promotion group size.
    """

    layout_version: int
    source_format_version: str
    sm_target: str
    group_size: int
    residual_scheme: str
    rounding_mode: str
    byte_order: str
    global_layout: str
    w13_layout: str
    logical_shape: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    nibble_order: str
    alpha_scope: str
    expert_mapping: tuple[int, ...]
    checksums: NVFP4V3Checksums

    def __post_init__(self) -> None:
        if type(self.layout_version) is not int:
            raise TypeError("layout_version must be an integer")
        if self.layout_version != NVFP4_SM90_LAYOUT_VERSION:
            raise ValueError(
                "NVFP4 layout version mismatch: expected "
                f"{NVFP4_SM90_LAYOUT_VERSION}, got {self.layout_version}"
            )
        if (
            not isinstance(self.source_format_version, str)
            or not self.source_format_version
        ):
            raise ValueError("source_format_version must be a non-empty string")
        if self.source_format_version != self.source_format_version.strip():
            raise ValueError(
                "source_format_version must not have surrounding whitespace"
            )
        if not isinstance(self.sm_target, str):
            raise TypeError("sm_target must be a string")
        if self.sm_target not in ("sm90", "sm90a"):
            raise ValueError("sm_target must be 'sm90' or 'sm90a'")
        if type(self.group_size) is not int:
            raise TypeError("group_size must be an integer")
        if self.group_size not in NVFP4_SM90_GROUP_SIZES:
            raise ValueError(
                f"group_size must be one of {NVFP4_SM90_GROUP_SIZES}, "
                f"got {self.group_size}"
            )
        if self.residual_scheme not in NVFP4_SM90_RESIDUAL_SCHEMES:
            raise ValueError(
                f"residual_scheme must be one of {NVFP4_SM90_RESIDUAL_SCHEMES}, "
                f"got {self.residual_scheme!r}"
            )
        if self.rounding_mode != NVFP4_SM90_ROUNDING_MODE:
            raise ValueError(f"rounding_mode must be {NVFP4_SM90_ROUNDING_MODE!r}")
        if not isinstance(self.byte_order, str):
            raise TypeError("byte_order must be a string")
        if self.byte_order != NVFP4_SM90_BYTE_ORDER:
            raise ValueError(f"byte_order must be {NVFP4_SM90_BYTE_ORDER!r}")
        if not isinstance(self.global_layout, str):
            raise TypeError("global_layout must be a string")
        if self.global_layout != NVFP4_SM90_GLOBAL_LAYOUT:
            raise ValueError(f"global_layout must be {NVFP4_SM90_GLOBAL_LAYOUT!r}")
        if not isinstance(self.w13_layout, str):
            raise TypeError("w13_layout must be a string")
        if self.w13_layout != NVFP4_SM90_W13_LAYOUT:
            raise ValueError(f"w13_layout must be {NVFP4_SM90_W13_LAYOUT!r}")
        if self.nibble_order != NVFP4_SM90_NIBBLE_ORDER:
            raise ValueError(f"nibble_order must be {NVFP4_SM90_NIBBLE_ORDER!r}")
        if self.alpha_scope not in NVFP4_SM90_ALPHA_SCOPES:
            raise ValueError(f"alpha_scope must be one of {NVFP4_SM90_ALPHA_SCOPES}")

        if (
            type(self.logical_shape) is not tuple
            or type(self.padded_shape) is not tuple
        ):
            raise TypeError("logical_shape and padded_shape must be tuples")
        if len(self.logical_shape) != 3 or len(self.padded_shape) != 3:
            raise ValueError(
                "logical_shape and padded_shape must have three dimensions"
            )
        if any(type(size) is not int for size in self.logical_shape):
            raise TypeError("logical_shape dimensions must be integers")
        if any(type(size) is not int for size in self.padded_shape):
            raise TypeError("padded_shape dimensions must be integers")
        logical = self.logical_shape
        padded = self.padded_shape
        experts, rows, columns = logical
        if experts < 0 or rows <= 0 or columns <= 0:
            raise ValueError(f"logical_shape must be [E>=0,N>0,K>0], got {logical}")
        expected_padded = (
            experts,
            math.ceil(rows / NVFP4_SM90_TILE_N) * NVFP4_SM90_TILE_N,
            math.ceil(columns / NVFP4_SM90_K_ALIGNMENT) * NVFP4_SM90_K_ALIGNMENT,
        )
        if padded != expected_padded:
            raise ValueError(
                "padded_shape must be minimal N64/K128 padding "
                f"{expected_padded}, got {padded}"
            )
        if type(self.expert_mapping) is not tuple:
            raise TypeError("expert_mapping must be a tuple")
        if any(type(expert) is not int for expert in self.expert_mapping):
            raise TypeError("expert_mapping entries must be integers")
        mapping = self.expert_mapping
        if len(mapping) != experts:
            raise ValueError(f"expert_mapping must contain E={experts} entries")
        if any(expert < 0 for expert in mapping) or len(set(mapping)) != len(mapping):
            raise ValueError("expert_mapping must contain unique non-negative entries")
        if not isinstance(self.checksums, NVFP4V3Checksums):
            raise TypeError("checksums must be NVFP4V3Checksums")

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "layout_version": self.layout_version,
            "source_format_version": self.source_format_version,
            "sm_target": self.sm_target,
            "group_size": self.group_size,
            "residual_scheme": self.residual_scheme,
            "rounding_mode": self.rounding_mode,
            "byte_order": self.byte_order,
            "global_layout": self.global_layout,
            "w13_layout": self.w13_layout,
            "logical_shape": list(self.logical_shape),
            "padded_shape": list(self.padded_shape),
            "nibble_order": self.nibble_order,
            "alpha_scope": self.alpha_scope,
            "expert_mapping": list(self.expert_mapping),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.metadata_dict(), "checksums": self.checksums.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NVFP4V3Manifest":
        if not isinstance(value, Mapping):
            raise TypeError("manifest must be a mapping")
        expected = {
            "layout_version",
            "source_format_version",
            "sm_target",
            "group_size",
            "residual_scheme",
            "rounding_mode",
            "byte_order",
            "global_layout",
            "w13_layout",
            "logical_shape",
            "padded_shape",
            "nibble_order",
            "alpha_scope",
            "expert_mapping",
            "checksums",
        }
        _require_exact_keys("manifest", value, expected)
        return cls(
            layout_version=value["layout_version"],
            source_format_version=value["source_format_version"],
            sm_target=value["sm_target"],
            group_size=value["group_size"],
            residual_scheme=value["residual_scheme"],
            rounding_mode=value["rounding_mode"],
            byte_order=value["byte_order"],
            global_layout=value["global_layout"],
            w13_layout=value["w13_layout"],
            logical_shape=tuple(value["logical_shape"]),
            padded_shape=tuple(value["padded_shape"]),
            nibble_order=value["nibble_order"],
            alpha_scope=value["alpha_scope"],
            expert_mapping=tuple(value["expert_mapping"]),
            checksums=NVFP4V3Checksums.from_dict(value["checksums"]),
        )


@dataclass(frozen=True)
class NVFP4SM90WeightViewV3:
    """Contiguous global-memory ABI for SM90 NVFP4 layout version 3.

    For a logical weight coordinate ``(e, n, k)`` and promotion group size
    ``G``, the four logical-to-physical mappings are::

        packed_e2m1[e, k//32, n//64, n%64, (k%32)//2]
        scale_e4m3_per16[e, k//32, n//64, n%64, (k%32)//16]
        promotion_residual[e, k//32, n//64, n%64, (k%32)//16]
        promotion_group_scale[e, k//G, n//64, n%64]

    The payload byte stores even ``k`` in its low nibble and odd ``k`` in its
    high nibble. ``N`` is minimally padded to 64 and ``K`` is minimally padded
    to 128, so every view has an integer number of K128 consumer stages for all
    supported promotion group sizes. Padded payload nibbles and per-16 scales are positive zero;
    per-16 blocks beyond ``ceil(K/16)`` use positive-zero BF16 residuals for
    ``generic`` or the -128 zero sentinel for ``pow2``. A partial final per-16
    block keeps its logical scale/residual while its padded payload nibbles are
    zero. Promotion group scales remain positive for all-zero padded groups;
    zero payload/residual values make those groups inert. Consumers crop
    results to ``logical_shape``.

    Every tensor uses standard C-contiguous storage in the shapes validated by
    :func:`validate_nvfp4_sm90_v3_layout`. Multi-byte scalar storage and
    checksum bytes are little-endian. Global memory has no swizzle. Any
    shared-memory swizzle is defined by the consuming kernel and is not part of
    this global-memory layout.
    """

    packed_e2m1: torch.Tensor
    scale_e4m3_per16: torch.Tensor
    promotion_group_scale: torch.Tensor
    promotion_residual: torch.Tensor
    global_alpha: torch.Tensor
    manifest: NVFP4V3Manifest

    def __post_init__(self) -> None:
        self.validate_layout()
        _validate_nvfp4_sm90_v3_values(self)

    def validate_layout(self) -> None:
        """Validate the structural ABI without hashing tensor contents."""

        validate_nvfp4_sm90_v3_layout(self)

    def verify_checksums(self) -> None:
        self.validate_layout()
        _validate_nvfp4_sm90_v3_values(self)
        actual = _v3_checksums(
            self.manifest.metadata_dict(),
            self.packed_e2m1,
            self.scale_e4m3_per16,
            self.promotion_group_scale,
            self.promotion_residual,
            self.global_alpha,
        )
        expected = self.manifest.checksums
        for name in (
            "payload_sha256",
            "scale_sha256",
            "group_scale_sha256",
            "residual_sha256",
            "alpha_sha256",
        ):
            if getattr(actual, name) != getattr(expected, name):
                tensor_name = name.removesuffix("_sha256")
                raise ValueError(f"NVFP4 v3 checksum mismatch for {tensor_name}")


def validate_nvfp4_sm90_v3_layout(view: NVFP4SM90WeightViewV3) -> None:
    """Validate every structural invariant consumed by the v3 layout ABI."""

    if not isinstance(view, NVFP4SM90WeightViewV3):
        raise TypeError("view must be NVFP4SM90WeightViewV3")
    if not isinstance(view.manifest, NVFP4V3Manifest):
        raise TypeError("manifest must be NVFP4V3Manifest")
    if sys.byteorder != view.manifest.byte_order:
        raise RuntimeError(
            "NVFP4 v3 tensors support only the manifest's little-endian host byte order"
        )

    tensors = {
        "packed_e2m1": view.packed_e2m1,
        "scale_e4m3_per16": view.scale_e4m3_per16,
        "promotion_group_scale": view.promotion_group_scale,
        "promotion_residual": view.promotion_residual,
        "global_alpha": view.global_alpha,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"v3 {name} must be a torch.Tensor")

    if view.packed_e2m1.dtype != torch.uint8 or view.packed_e2m1.ndim != 5:
        raise ValueError("v3 payload must be contiguous uint8 [E,Kt,Nt,64,16]")
    if (
        view.scale_e4m3_per16.dtype != torch.float8_e4m3fn
        or view.scale_e4m3_per16.ndim != 5
    ):
        raise ValueError("v3 scales must be contiguous E4M3FN [E,Kt,Nt,64,2]")
    if (
        view.promotion_group_scale.dtype != torch.float32
        or view.promotion_group_scale.ndim != 4
    ):
        raise ValueError(
            "v3 promotion_group_scale must be contiguous float32 [E,Kg,Nt,64]"
        )
    expected_residual_dtype = (
        torch.bfloat16 if view.manifest.residual_scheme == "generic" else torch.int8
    )
    if (
        view.promotion_residual.dtype != expected_residual_dtype
        or view.promotion_residual.ndim != 5
    ):
        raise ValueError(
            "v3 promotion_residual must be contiguous BF16 for generic or "
            "int8 exponents for pow2 [E,Kt,Nt,64,2]"
        )
    if view.global_alpha.dtype != torch.float32 or view.global_alpha.ndim > 1:
        raise ValueError("v3 global_alpha must be scalar or contiguous float32 [E]")
    if not all(tensor.is_contiguous() for tensor in tensors.values()):
        raise ValueError("v3 tensors must be contiguous")
    if len({tensor.device for tensor in tensors.values()}) != 1:
        raise ValueError("v3 tensors must share a device")

    experts, padded_n, padded_k = view.manifest.padded_shape
    expected_payload = (
        experts,
        padded_k // NVFP4_SM90_TILE_K,
        padded_n // NVFP4_SM90_TILE_N,
        NVFP4_SM90_TILE_N,
        NVFP4_SM90_TILE_K // 2,
    )
    expected_scales = expected_payload[:-1] + (NVFP4_SM90_TILE_K // 16,)
    expected_group_scales = (
        experts,
        padded_k // view.manifest.group_size,
        padded_n // NVFP4_SM90_TILE_N,
        NVFP4_SM90_TILE_N,
    )
    expected_shapes = {
        "payload": (view.packed_e2m1, expected_payload),
        "scale": (view.scale_e4m3_per16, expected_scales),
        "promotion group scale": (
            view.promotion_group_scale,
            expected_group_scales,
        ),
        "promotion residual": (view.promotion_residual, expected_scales),
    }
    for name, (tensor, expected_shape) in expected_shapes.items():
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"v3 {name} shape must be {expected_shape}, got {tuple(tensor.shape)}"
            )
    expected_alpha_shape = (
        () if view.manifest.alpha_scope == "per_tensor" else (experts,)
    )
    if tuple(view.global_alpha.shape) != expected_alpha_shape:
        raise ValueError(
            f"v3 global_alpha shape must be {expected_alpha_shape} for "
            f"{view.manifest.alpha_scope}"
        )


def _iter_validation_chunks(tensor: torch.Tensor):
    flat = tensor.detach().reshape(-1)
    for begin in range(0, flat.numel(), _NVFP4_SM90_VALIDATION_CHUNK_ELEMENTS):
        yield flat[begin : begin + _NVFP4_SM90_VALIDATION_CHUNK_ELEMENTS]


def _validate_e4m3_scale_values(scales: torch.Tensor) -> None:
    for chunk in _iter_validation_chunks(scales):
        raw = chunk.view(torch.uint8)
        if bool((raw.bitwise_and(0x7F) == 0x7F).any()):
            raise ValueError("v3 scales must be finite")
        if bool((raw == 0x80).any()):
            raise ValueError("v3 scales must not contain negative zero")
        if bool((raw.bitwise_and(0x80) != 0).any()):
            raise ValueError("v3 scales must be non-negative")


def _validate_floating_values(
    tensor: torch.Tensor,
    *,
    name: str,
    positive: bool,
    reject_bfloat16_negative_zero: bool = False,
) -> None:
    for chunk in _iter_validation_chunks(tensor):
        if reject_bfloat16_negative_zero and bool(
            (chunk.view(torch.int16) == -0x8000).any()
        ):
            raise ValueError(f"{name} must not contain negative zero")
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError(f"{name} must be finite")
        invalid = (chunk <= 0).any() if positive else (chunk < 0).any()
        if bool(invalid):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be {qualifier}")


def _validate_nvfp4_sm90_v3_values(view: NVFP4SM90WeightViewV3) -> None:
    _validate_e4m3_scale_values(view.scale_e4m3_per16)
    _validate_floating_values(
        view.global_alpha,
        name="v3 global_alpha",
        positive=True,
    )
    _validate_floating_values(
        view.promotion_group_scale,
        name="v3 promotion group scales",
        positive=True,
    )
    if view.manifest.residual_scheme == "generic":
        _validate_floating_values(
            view.promotion_residual,
            name="v3 generic promotion residuals",
            positive=False,
            reject_bfloat16_negative_zero=True,
        )


@dataclass(frozen=True)
class NVFP4SM90WeightBundleV3:
    """Offline policy bundle; W4A16 fallback remains a canonical checkpoint."""

    promoted_buckets: tuple[NVFP4SM90WeightViewV3, ...]
    w4a16_fallback: NVFP4Checkpoint | None
    expert_mapping: tuple[int, ...]
    residual_scheme: str

    def __post_init__(self) -> None:
        buckets = tuple(self.promoted_buckets)
        if not isinstance(self.expert_mapping, Sequence) or isinstance(
            self.expert_mapping, (str, bytes)
        ):
            raise TypeError("expert_mapping must be a sequence")
        if any(type(expert) is not int for expert in self.expert_mapping):
            raise TypeError("expert_mapping entries must be integers")
        mapping = tuple(self.expert_mapping)
        if self.residual_scheme not in NVFP4_SM90_RESIDUAL_SCHEMES:
            raise ValueError(
                f"residual_scheme must be one of {NVFP4_SM90_RESIDUAL_SCHEMES}"
            )
        if len(mapping) != len(set(mapping)) or any(expert < 0 for expert in mapping):
            raise ValueError("expert_mapping must contain unique non-negative entries")
        bucket_groups = []
        covered: list[int] = []
        for bucket in buckets:
            if not isinstance(bucket, NVFP4SM90WeightViewV3):
                raise TypeError("promoted_buckets must contain v3 weight views")
            if bucket.manifest.residual_scheme != self.residual_scheme:
                raise ValueError("all promoted buckets must use residual_scheme")
            bucket_groups.append(bucket.manifest.group_size)
            covered.extend(bucket.manifest.expert_mapping)
        if len(bucket_groups) != len(set(bucket_groups)):
            raise ValueError("promoted buckets must have unique group sizes")
        if self.w4a16_fallback is not None:
            if not isinstance(self.w4a16_fallback, NVFP4Checkpoint):
                raise TypeError("w4a16_fallback must be an NVFP4Checkpoint")
            covered.extend(self.w4a16_fallback.expert_mapping)
        if len(covered) != len(set(covered)) or set(covered) != set(mapping):
            raise ValueError(
                "promoted buckets and W4A16 fallback must partition expert_mapping"
            )
        object.__setattr__(self, "promoted_buckets", buckets)
        object.__setattr__(self, "expert_mapping", mapping)

    def verify_checksums(self) -> None:
        """Verify promoted v3 buckets; canonical fallback has no v3 manifest."""

        for bucket in self.promoted_buckets:
            bucket.verify_checksums()


def _zero_e4m3(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.zeros(tuple(shape), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )


def _v3_metadata(
    checkpoint: NVFP4Checkpoint,
    *,
    sm_target: str,
    group_size: int,
    residual_scheme: str,
    rounding_mode: str,
    padded_shape: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        "layout_version": NVFP4_SM90_LAYOUT_VERSION,
        "source_format_version": checkpoint.source_format_version,
        "sm_target": sm_target,
        "group_size": group_size,
        "residual_scheme": residual_scheme,
        "rounding_mode": rounding_mode,
        "byte_order": NVFP4_SM90_BYTE_ORDER,
        "global_layout": NVFP4_SM90_GLOBAL_LAYOUT,
        "w13_layout": NVFP4_SM90_W13_LAYOUT,
        "logical_shape": list(checkpoint.logical_shape),
        "padded_shape": list(padded_shape),
        "nibble_order": NVFP4_SM90_NIBBLE_ORDER,
        "alpha_scope": checkpoint.alpha_scope,
        "expert_mapping": list(checkpoint.expert_mapping),
    }


def _v3_checksums(
    metadata: Mapping[str, Any],
    payload: torch.Tensor,
    scales: torch.Tensor,
    group_scales: torch.Tensor,
    residuals: torch.Tensor,
    alpha: torch.Tensor,
) -> NVFP4V3Checksums:
    return NVFP4V3Checksums(
        payload_sha256=_tensor_sha256(
            payload, tensor_name="payload", metadata=metadata
        ),
        scale_sha256=_tensor_sha256(scales, tensor_name="scale", metadata=metadata),
        group_scale_sha256=_tensor_sha256(
            group_scales, tensor_name="group_scale", metadata=metadata
        ),
        residual_sha256=_tensor_sha256(
            residuals, tensor_name="residual", metadata=metadata
        ),
        alpha_sha256=_tensor_sha256(alpha, tensor_name="alpha", metadata=metadata),
    )


def _decode_linear_e2m1(payload: torch.Tensor) -> torch.Tensor:
    low = payload.bitwise_and(0x0F)
    high = payload.bitwise_right_shift(4).bitwise_and(0x0F)
    codes = torch.stack((low, high), dim=-1).reshape(
        *payload.shape[:-1], payload.shape[-1] * 2
    )
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=payload.device,
    )
    decoded = magnitudes[(codes & 0x07).to(torch.int64)]
    return torch.where((codes & 0x08) != 0, -decoded, decoded)


def _materialize_promotion_streams(
    linear_payload: torch.Tensor,
    linear_scales: torch.Tensor,
    *,
    group_size: int,
    residual_scheme: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts, padded_n, scale_blocks = linear_scales.shape
    padded_k = scale_blocks * 16
    groups = padded_k // group_size
    blocks_per_group = group_size // 16
    total_rows = experts * padded_n
    flat_payload = linear_payload.view(total_rows, padded_k // 2)
    flat_scales = linear_scales.view(total_rows, scale_blocks)
    group_scale_output = torch.empty(
        (total_rows, groups), dtype=torch.float32, device=linear_payload.device
    )
    residual_dtype = torch.bfloat16 if residual_scheme == "generic" else torch.int8
    residual_output = torch.empty(
        (total_rows, scale_blocks),
        dtype=residual_dtype,
        device=linear_payload.device,
    )
    rows_per_chunk = max(1, _NVFP4_SM90_PROMOTION_CHUNK_ELEMENTS // padded_k)
    for begin in range(0, total_rows, rows_per_chunk):
        end = min(begin + rows_per_chunk, total_rows)
        row_count = end - begin
        codes = (
            _decode_linear_e2m1(flat_payload[begin:end])
            .abs()
            .reshape(row_count, groups, blocks_per_group, 16)
        )
        code_block_max = codes.amax(dim=-1)
        scales = (
            flat_scales[begin:end]
            .to(torch.float32)
            .reshape(row_count, groups, blocks_per_group)
        )
        group_scale = (code_block_max * scales).amax(dim=-1) / NVFP4_SM90_E4M3_MAX
        group_scale = torch.where(
            group_scale > 0, group_scale, torch.ones_like(group_scale)
        )

        if residual_scheme == "generic":
            group_scale = group_scale * NVFP4_SM90_GENERIC_MARGIN
            residual = (scales / group_scale.unsqueeze(-1)).to(torch.bfloat16)
        else:
            ratio = scales / group_scale.unsqueeze(-1)
            exponent = torch.where(
                ratio > 0,
                torch.round(torch.log2(ratio)),
                torch.zeros_like(ratio),
            ).clamp(NVFP4_SM90_POW2_ZERO_SENTINEL + 1, 127)
            residual_value = torch.where(
                ratio > 0,
                torch.pow(torch.full_like(exponent, 2.0), exponent),
                torch.zeros_like(exponent),
            )
            normalized_max = (code_block_max * residual_value).amax(dim=-1)
            shift = torch.where(
                normalized_max > NVFP4_SM90_E4M3_MAX,
                torch.ceil(torch.log2(normalized_max / NVFP4_SM90_E4M3_MAX)),
                torch.zeros_like(normalized_max),
            ).clamp_min(0)
            group_scale = group_scale * torch.pow(torch.full_like(shift, 2.0), shift)
            exponent = (exponent - shift.unsqueeze(-1)).clamp(
                NVFP4_SM90_POW2_ZERO_SENTINEL + 1, 127
            )
            residual = torch.where(
                ratio > 0,
                exponent,
                torch.full_like(exponent, NVFP4_SM90_POW2_ZERO_SENTINEL),
            ).to(torch.int8)
        group_scale_output[begin:end] = group_scale
        residual_output[begin:end] = residual.reshape(row_count, scale_blocks)
    return group_scale_output.view(experts, padded_n, groups), residual_output.view(
        experts, padded_n, scale_blocks
    )


@torch.no_grad()
def repack_nvfp4_sm90_v3(
    checkpoint: NVFP4Checkpoint,
    *,
    group_size: int,
    residual_scheme: str,
    sm_target: str = "sm90a",
    rounding_mode: str = NVFP4_SM90_ROUNDING_MODE,
) -> NVFP4SM90WeightViewV3:
    """Build deterministic K-major K32/N64 storage and W4A8 promotion streams.

    The exact per-16 E4M3FN stream is retained for lossless recovery.  Generic
    promotion stores BF16 residuals; pow2 promotion stores signed int8
    exponents with -128 as the zero sentinel. K is minimally padded to 128 for
    the consumer stage shape. Global memory is contiguous and unswizzled; a
    consuming kernel defines any shared-memory swizzle.
    """

    if not isinstance(checkpoint, NVFP4Checkpoint):
        raise TypeError("checkpoint must be NVFP4Checkpoint")
    if group_size not in NVFP4_SM90_GROUP_SIZES:
        raise ValueError(f"group_size must be one of {NVFP4_SM90_GROUP_SIZES}")
    if residual_scheme not in NVFP4_SM90_RESIDUAL_SCHEMES:
        raise ValueError(
            f"residual_scheme must be one of {NVFP4_SM90_RESIDUAL_SCHEMES}, "
            f"got {residual_scheme!r}"
        )
    experts, logical_n, logical_k = checkpoint.logical_shape
    padded_n = math.ceil(logical_n / NVFP4_SM90_TILE_N) * NVFP4_SM90_TILE_N
    padded_k = math.ceil(logical_k / NVFP4_SM90_K_ALIGNMENT) * NVFP4_SM90_K_ALIGNMENT

    linear_payload = torch.zeros(
        (experts, padded_n, padded_k // 2),
        dtype=torch.uint8,
        device=checkpoint.device,
    )
    logical_payload_bytes = math.ceil(logical_k / 2)
    linear_payload[:, :logical_n, :logical_payload_bytes] = checkpoint.packed_e2m1[
        :, :logical_n, :logical_payload_bytes
    ]
    if logical_k % 2:
        linear_payload[:, :logical_n, logical_payload_bytes - 1].bitwise_and_(0x0F)

    linear_scales = _zero_e4m3((experts, padded_n, padded_k // 16), checkpoint.device)
    logical_scale_blocks = math.ceil(logical_k / 16)
    linear_scales[:, :logical_n, :logical_scale_blocks] = checkpoint.scale_e4m3_per16[
        :, :logical_n, :logical_scale_blocks
    ]

    n_tiles = padded_n // NVFP4_SM90_TILE_N
    k_tiles = padded_k // NVFP4_SM90_TILE_K
    payload = (
        linear_payload.view(
            experts,
            n_tiles,
            NVFP4_SM90_TILE_N,
            k_tiles,
            NVFP4_SM90_TILE_K // 2,
        )
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    scales = (
        linear_scales.view(
            experts,
            n_tiles,
            NVFP4_SM90_TILE_N,
            k_tiles,
            NVFP4_SM90_TILE_K // 16,
        )
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    linear_group_scales, linear_residuals = _materialize_promotion_streams(
        linear_payload,
        linear_scales,
        group_size=group_size,
        residual_scheme=residual_scheme,
    )
    group_scales = (
        linear_group_scales.view(
            experts,
            n_tiles,
            NVFP4_SM90_TILE_N,
            padded_k // group_size,
        )
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    residuals = (
        linear_residuals.view(
            experts,
            n_tiles,
            NVFP4_SM90_TILE_N,
            k_tiles,
            NVFP4_SM90_TILE_K // 16,
        )
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    alpha = checkpoint.global_alpha.contiguous()
    padded_shape = (experts, padded_n, padded_k)
    metadata = _v3_metadata(
        checkpoint,
        sm_target=sm_target,
        group_size=group_size,
        residual_scheme=residual_scheme,
        rounding_mode=rounding_mode,
        padded_shape=padded_shape,
    )
    manifest = NVFP4V3Manifest(
        layout_version=NVFP4_SM90_LAYOUT_VERSION,
        source_format_version=checkpoint.source_format_version,
        sm_target=sm_target,
        group_size=group_size,
        residual_scheme=residual_scheme,
        rounding_mode=rounding_mode,
        byte_order=NVFP4_SM90_BYTE_ORDER,
        global_layout=NVFP4_SM90_GLOBAL_LAYOUT,
        w13_layout=NVFP4_SM90_W13_LAYOUT,
        logical_shape=checkpoint.logical_shape,
        padded_shape=padded_shape,
        nibble_order=NVFP4_SM90_NIBBLE_ORDER,
        alpha_scope=checkpoint.alpha_scope,
        expert_mapping=checkpoint.expert_mapping,
        checksums=_v3_checksums(
            metadata, payload, scales, group_scales, residuals, alpha
        ),
    )
    return NVFP4SM90WeightViewV3(
        payload, scales, group_scales, residuals, alpha, manifest
    )


def _slice_nvfp4_checkpoint(
    checkpoint: NVFP4Checkpoint, expert_indices: Sequence[int]
) -> NVFP4Checkpoint:
    indices = tuple(int(index) for index in expert_indices)
    index = torch.tensor(indices, dtype=torch.int64, device=checkpoint.device)
    alpha = (
        checkpoint.global_alpha
        if checkpoint.alpha_scope == "per_tensor"
        else checkpoint.global_alpha.index_select(0, index).contiguous()
    )
    return NVFP4Checkpoint(
        checkpoint.packed_e2m1.index_select(0, index).contiguous(),
        checkpoint.scale_e4m3_per16.index_select(0, index).contiguous(),
        alpha,
        (len(indices), checkpoint.logical_shape[1], checkpoint.logical_shape[2]),
        tuple(checkpoint.expert_mapping[expert] for expert in indices),
        checkpoint.source_format_version,
    )


@torch.no_grad()
def repack_nvfp4_sm90_v3_selected(
    checkpoint: NVFP4Checkpoint,
    selection: Mapping[str, Any],
    *,
    sm_target: str = "sm90a",
    rounding_mode: str = NVFP4_SM90_ROUNDING_MODE,
) -> NVFP4SM90WeightBundleV3:
    """Materialize a scale-lab policy as homogeneous per-G expert buckets.

    Experts selected for W4A16 remain in canonical checkpoint form for the
    existing W4A16 preparation path; they are not runtime-packed by this API.
    """

    if not isinstance(checkpoint, NVFP4Checkpoint):
        raise TypeError("checkpoint must be NVFP4Checkpoint")
    if not isinstance(selection, Mapping):
        raise TypeError("selection must be a mapping")
    mode = selection.get("mode")
    if mode not in ("model", "per_expert"):
        raise ValueError("selection mode must be 'model' or 'per_expert'")
    residual_scheme = selection.get("residual_scheme")
    if residual_scheme not in NVFP4_SM90_RESIDUAL_SCHEMES:
        raise ValueError(
            f"selection residual_scheme must be one of {NVFP4_SM90_RESIDUAL_SCHEMES}"
        )
    entries = selection.get("experts")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("selection experts must be a sequence")
    choices: dict[int, int | None] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or "expert_id" not in entry:
            raise TypeError("each selection expert must be a mapping with expert_id")
        expert_id = entry["expert_id"]
        if type(expert_id) is not int:
            raise TypeError("selection expert_id must be an integer")
        if expert_id in choices:
            raise ValueError(f"selection repeats expert_id {expert_id}")
        group_size = entry.get("group_size")
        if group_size is not None:
            if type(group_size) is not int:
                raise TypeError("selection group_size must be an integer or None")
            if group_size not in NVFP4_SM90_GROUP_SIZES:
                raise ValueError(
                    f"selection group_size must be one of {NVFP4_SM90_GROUP_SIZES}"
                )
        choices[expert_id] = group_size
    expected_experts = set(checkpoint.expert_mapping)
    if set(choices) != expected_experts:
        missing = sorted(expected_experts - set(choices))
        extra = sorted(set(choices) - expected_experts)
        raise ValueError(
            f"selection must cover checkpoint experts exactly: missing={missing}, "
            f"extra={extra}"
        )
    if mode == "model":
        model_group = selection.get("group_size")
        if type(model_group) is not int:
            raise TypeError("model selection group_size must be an integer")
        if model_group not in NVFP4_SM90_GROUP_SIZES or any(
            group_size != model_group for group_size in choices.values()
        ):
            raise ValueError(
                "model selection must assign its group_size to every expert"
            )

    buckets = []
    for group_size in NVFP4_SM90_GROUP_SIZES:
        indices = [
            index
            for index, expert_id in enumerate(checkpoint.expert_mapping)
            if choices[expert_id] == group_size
        ]
        if indices:
            buckets.append(
                repack_nvfp4_sm90_v3(
                    _slice_nvfp4_checkpoint(checkpoint, indices),
                    group_size=group_size,
                    residual_scheme=residual_scheme,
                    sm_target=sm_target,
                    rounding_mode=rounding_mode,
                )
            )
    fallback_indices = [
        index
        for index, expert_id in enumerate(checkpoint.expert_mapping)
        if choices[expert_id] is None
    ]
    fallback = (
        _slice_nvfp4_checkpoint(checkpoint, fallback_indices)
        if fallback_indices
        else None
    )
    return NVFP4SM90WeightBundleV3(
        tuple(buckets), fallback, checkpoint.expert_mapping, residual_scheme
    )


@torch.no_grad()
def unpack_nvfp4_sm90_v3(
    view: NVFP4SM90WeightViewV3, *, verify_checksums: bool = True
) -> NVFP4Checkpoint:
    if not isinstance(view, NVFP4SM90WeightViewV3):
        raise TypeError("view must be NVFP4SM90WeightViewV3")
    if verify_checksums:
        view.verify_checksums()
    else:
        view.validate_layout()
        _validate_nvfp4_sm90_v3_values(view)
    experts, padded_n, padded_k = view.manifest.padded_shape
    payload = (
        view.packed_e2m1.permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(experts, padded_n, padded_k // 2)
    )
    scales = (
        view.scale_e4m3_per16.permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(experts, padded_n, padded_k // 16)
    )
    return NVFP4Checkpoint(
        packed_e2m1=payload,
        scale_e4m3_per16=scales,
        global_alpha=view.global_alpha,
        logical_shape=view.manifest.logical_shape,
        expert_mapping=view.manifest.expert_mapping,
        source_format_version=view.manifest.source_format_version,
    )


def _cast_promoted_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Apply the finite, saturating E4M3FN promotion conversion."""

    for chunk in _iter_validation_chunks(values):
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("promoted E4M3 inputs must be finite")
    return values.clamp(-NVFP4_SM90_E4M3_MAX, NVFP4_SM90_E4M3_MAX).to(
        torch.float8_e4m3fn
    )


@torch.no_grad()
def reference_dequantize_nvfp4_sm90_v3_promoted(
    view: NVFP4SM90WeightViewV3,
    *,
    apply_global_alpha: bool = True,
    verify_checksums: bool = True,
) -> torch.Tensor:
    """Decode the materialized W4A8 operand stream in FP32 for validation."""

    if not isinstance(view, NVFP4SM90WeightViewV3):
        raise TypeError("view must be NVFP4SM90WeightViewV3")
    if verify_checksums:
        view.verify_checksums()
    else:
        view.validate_layout()
        _validate_nvfp4_sm90_v3_values(view)
    experts, padded_n, padded_k = view.manifest.padded_shape
    payload = (
        view.packed_e2m1.permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(experts, padded_n, padded_k // 2)
    )
    residual = (
        view.promotion_residual.permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(experts, padded_n, padded_k // 16)
    )
    group_scale = (
        view.promotion_group_scale.permute(0, 2, 3, 1)
        .contiguous()
        .view(experts, padded_n, padded_k // view.manifest.group_size)
    )
    if view.manifest.residual_scheme == "generic":
        residual_value = residual.to(torch.float32)
    else:
        exponent = residual.to(torch.float32)
        residual_value = torch.where(
            residual == NVFP4_SM90_POW2_ZERO_SENTINEL,
            torch.zeros_like(exponent),
            torch.pow(torch.full_like(exponent, 2.0), exponent),
        )
    normalized = _decode_linear_e2m1(payload) * residual_value.repeat_interleave(
        16, dim=-1
    )
    promoted = _cast_promoted_e4m3(normalized).to(torch.float32)
    promoted = promoted * group_scale.repeat_interleave(
        view.manifest.group_size, dim=-1
    )
    if apply_global_alpha:
        alpha = (
            view.global_alpha.expand(experts)
            if view.global_alpha.ndim == 0
            else view.global_alpha
        )
        promoted = promoted * alpha[:, None, None]
    _, logical_n, logical_k = view.manifest.logical_shape
    return promoted[:, :logical_n, :logical_k].contiguous()


@torch.no_grad()
def convert_nvfp4_rs_v2_to_v3(
    view: NVFP4RSWeightView,
    *,
    source_layout_version: int,
    logical_shape: Sequence[int],
    expert_mapping: Sequence[int],
    source_format_version: str,
    alpha_scope: str,
    group_size: int,
    residual_scheme: str,
    sm_target: str = "sm90a",
) -> NVFP4SM90WeightViewV3:
    if source_layout_version != NVFP4_RS_LAYOUT_VERSION:
        raise ValueError(
            "NVFP4 source layout version mismatch: expected "
            f"{NVFP4_RS_LAYOUT_VERSION}, got {source_layout_version}"
        )
    if not isinstance(view, NVFP4RSWeightView):
        raise TypeError("view must be NVFP4RSWeightView")
    if alpha_scope not in NVFP4_SM90_ALPHA_SCOPES:
        raise ValueError(f"alpha_scope must be one of {NVFP4_SM90_ALPHA_SCOPES}")
    alpha = view.alpha
    if alpha_scope == "per_tensor":
        if alpha.numel() == 0:
            raise ValueError("an empty v2 view cannot recover a per-tensor alpha")
        if not bool((alpha == alpha[0]).all()):
            raise ValueError(
                "v2 alpha values differ and cannot be converted to per-tensor scope"
            )
        alpha = alpha[0].reshape(())
    if len(logical_shape) != 3:
        raise ValueError(
            f"logical_shape must contain three dimensions, got {len(logical_shape)}"
        )
    checkpoint = NVFP4Checkpoint(
        packed_e2m1=unpack_nvfp4_payload_v2(view.payload),
        scale_e4m3_per16=unpack_nvfp4_scales_v2(view.scales),
        global_alpha=alpha,
        logical_shape=(
            int(logical_shape[0]),
            int(logical_shape[1]),
            int(logical_shape[2]),
        ),
        expert_mapping=tuple(expert_mapping),
        source_format_version=source_format_version,
    )
    return repack_nvfp4_sm90_v3(
        checkpoint,
        group_size=group_size,
        residual_scheme=residual_scheme,
        sm_target=sm_target,
    )
