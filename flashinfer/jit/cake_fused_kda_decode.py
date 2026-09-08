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

import functools
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from . import env as jit_env
from ._kda_jit_common import (
    gen_kda_jit_spec,
    get_flashinfer_include_dir,
    get_kda_csrc_dir,
)
from .core import JitSpec, logger, sm100a_nvcc_flags
from .utils import write_if_different

CakeFusedKDADecodeTarget = Literal["sm100a"]
CakeFusedKDADecodeStateIndicesMode = Literal[
    "positive_unique",
    "unique_or_null",
    "repeated_positive",
]

_BINDING_HEADER = "cake_fused_kda_decode_binding.cuh"
_TARGETS: tuple[CakeFusedKDADecodeTarget, ...] = ("sm100a",)
_STATE_INDICES_MODES: tuple[CakeFusedKDADecodeStateIndicesMode, ...] = (
    "positive_unique",
    "unique_or_null",
    "repeated_positive",
)
_TARGET_DEFINE = "-DFLASHINFER_CAKE_FUSED_KDA_DECODE_TARGET_MINOR=0"

_COMMON_BUFFER_ABI: tuple[tuple[str, str, str], ...] = (
    ("buffer", "x", "bfloat16"),
    ("buffer", "weight", "float32"),
    ("buffer", "conv_state", "bfloat16"),
    ("buffer", "raw_gate", "bfloat16"),
    ("buffer", "raw_beta", "bfloat16"),
    ("buffer", "A_log", "float32"),
    ("buffer", "dt_bias", "float32"),
    ("buffer", "state_indices", "int32"),
    ("buffer", "state", "state_dtype"),
    ("buffer", "output_gate", "bfloat16"),
    ("buffer", "norm_weight", "float32"),
    ("buffer", "output", "bfloat16"),
)
_COMMON_SCALAR_ABI: tuple[tuple[str, str, str], ...] = (
    ("parameter", "x_row_stride", "int32"),
    ("parameter", "conv_slot_stride", "int32"),
    ("parameter", "beta_row_stride", "int32"),
    ("parameter", "state_slot_stride", "int32"),
    ("parameter", "output_gate_row_stride", "int32"),
    ("parameter", "H", "int32"),
)
_RUNTIME_CONFIG_ABI: tuple[tuple[str, str, str], ...] = (
    ("parameter", "use_lower_bound", "int32"),
    ("parameter", "lower_bound_log2", "float32_scalar"),
    ("parameter", "norm_eps", "float32_scalar"),
)
CAKE_FUSED_KDA_DECODE_ABIS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "standard": _COMMON_BUFFER_ABI + _COMMON_SCALAR_ABI + _RUNTIME_CONFIG_ABI,
    "repeated_safe": (
        _COMMON_BUFFER_ABI
        + _COMMON_SCALAR_ABI
        + (("parameter", "rows", "int32"),)
        + _RUNTIME_CONFIG_ABI
    ),
}
_ARG_PLAN_SHA256 = {
    name: hashlib.sha256(
        json.dumps(arguments, separators=(",", ":")).encode()
    ).hexdigest()
    for name, arguments in CAKE_FUSED_KDA_DECODE_ABIS.items()
}


@dataclass(frozen=True)
class CakeFusedKDADecodeEligibility:
    """One conjunction of host-known facts selecting an exported Cake kernel."""

    heads: tuple[int, ...]
    minimum_rows: int
    maximum_rows: int | None
    state_indices_modes: tuple[CakeFusedKDADecodeStateIndicesMode, ...]
    lower_bound_values: tuple[float | None, ...] | None
    norm_eps_values: tuple[float, ...] | None
    strides: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True)
class CakeFusedKDADecodeVariant:
    """One explicitly registered Cake device source and launch description."""

    name: str
    target: CakeFusedKDADecodeTarget
    body_path: Path
    source_sha256: str
    kernel_symbol: str
    abi_kind: str
    state_dtype: str
    slot_offset_bits: int
    extra_cuda_cflags: tuple[str, ...]
    threads: int
    dynamic_smem_bytes: int
    eligibility: tuple[CakeFusedKDADecodeEligibility, ...]


# Target-owned static registration. There is deliberately no runtime generated
# manifest: the JIT loader is the source of truth for its checked-in closure.
_VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "repeated_safe_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_repeated_safe_f32_wide_slot_offsets.cu",
        "source_sha256": "5f78b057b6f95aadd5b866f0f07485db4f7bf2287407a013fdbbac3da0fd69db",
        "kernel_symbol": "kernel_cake_fused_kda_decode_repeated_safe_f32_wide_slot_offsets",
        "abi_kind": "repeated_safe",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "repeated_safe_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_repeated_safe_f32.cu",
        "source_sha256": "cc4dff48d4c0b3c5b50f73e57f394df7de1fd951af82614341ab029bf0f9c5a5",
        "kernel_symbol": "kernel_cake_fused_kda_decode_repeated_safe_f32",
        "abi_kind": "repeated_safe",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "repeated_safe_bf16_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_repeated_safe_bf16_wide_slot_offsets.cu",
        "source_sha256": "c092826928d4d3ba752bdb4f63a1766c457aa88c0bd999f1da2574346b6046fc",
        "kernel_symbol": "kernel_cake_fused_kda_decode_repeated_safe_bf16_wide_slot_offsets",
        "abi_kind": "repeated_safe",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "repeated_safe_bf16",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_repeated_safe_bf16.cu",
        "source_sha256": "669752084fe9a62427330158d6795add31be81f260f1a57b7e6b38b4cdba5c11",
        "kernel_symbol": "kernel_cake_fused_kda_decode_repeated_safe_bf16",
        "abi_kind": "repeated_safe",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": None,
                "state_indices_modes": ["repeated_positive"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "wide512_positive_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_wide512_positive_f32_wide_slot_offsets.cu",
        "source_sha256": "83c937bd1a82368424c6e9920e8e35ba86d07291f34c2685a6f349920cffa7cd",
        "kernel_symbol": "kernel_cake_fused_kda_decode_wide512_positive_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 512,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": 6,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": 3,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": 1,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "wide512_positive_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_wide512_positive_f32.cu",
        "source_sha256": "e8ed8c5274cb2d48f680815a3f69492155d117bf62d425db30df97a527b633cf",
        "kernel_symbol": "kernel_cake_fused_kda_decode_wide512_positive_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 512,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": 6,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": 3,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": 1,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "wide512_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_wide512_f32_wide_slot_offsets.cu",
        "source_sha256": "97072c8dd22b9e443e13c95c817ec32bf738f40d775bccf96185a7a4d3820a3c",
        "kernel_symbol": "kernel_cake_fused_kda_decode_wide512_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 512,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": 12,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": 6,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": 4,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": 3,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": 1,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "wide512_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_wide512_f32.cu",
        "source_sha256": "10148bbeb052b16272dcc9626e5b671c87ff79e8b1cf47cce86b07bbb24941cf",
        "kernel_symbol": "kernel_cake_fused_kda_decode_wide512_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 512,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": 12,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": 6,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": 4,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": 3,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": 1,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "wide512_bf16_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_wide512_bf16_wide_slot_offsets.cu",
        "source_sha256": "e9c4a072209d51c813bfebec696c7b28c2ae75d03707979857a8c4c249c132a3",
        "kernel_symbol": "kernel_cake_fused_kda_decode_wide512_bf16_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 512,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": 6,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": 3,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": 1,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "wide512_bf16",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_wide512_bf16.cu",
        "source_sha256": "1a5d0b73451251e43c6564a4449f556e464a92218c44bfb68488431fbdf55f9e",
        "kernel_symbol": "kernel_cake_fused_kda_decode_wide512_bf16",
        "abi_kind": "standard",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 512,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 1,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 1,
                "maximum_rows": 6,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 1,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 1,
                "maximum_rows": 3,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 1,
                "maximum_rows": 1,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "compact_async_pr_eval_h96_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_compact_async_pr_eval_h96_f32_wide_slot_offsets.cu",
        "source_sha256": "d54e9f2f0f87eafc777e5b2d49d99952abf340900c0aa0f20f978bdbb9731be3",
        "kernel_symbol": "kernel_cake_fused_kda_decode_compact_async_pr_eval_h96_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 36480,
        "eligibility": (
            {
                "heads": [96],
                "minimum_rows": 4,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 36881,
                    "conv_slot_stride": 3256320,
                    "beta_row_stride": 97,
                    "state_slot_stride": 1628160,
                    "output_gate_row_stride": 12295,
                },
            },
        ),
    },
    {
        "name": "compact_async_pr_eval_h96_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_compact_async_pr_eval_h96_f32.cu",
        "source_sha256": "9c25c4a66e2d4d16448cf6c9ed55430b4ce2aca59c76cf1a8e64198b64bd758c",
        "kernel_symbol": "kernel_cake_fused_kda_decode_compact_async_pr_eval_h96_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 36480,
        "eligibility": (
            {
                "heads": [96],
                "minimum_rows": 4,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 36881,
                    "conv_slot_stride": 3256320,
                    "beta_row_stride": 97,
                    "state_slot_stride": 1628160,
                    "output_gate_row_stride": 12295,
                },
            },
        ),
    },
    {
        "name": "compact_async_positive_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_compact_async_positive_f32_wide_slot_offsets.cu",
        "source_sha256": "13a7c857149af111d98aba0c6e9f19ba36d382318ebba5c40ec77d0be4111a58",
        "kernel_symbol": "kernel_cake_fused_kda_decode_compact_async_positive_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 36480,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 25,
                "maximum_rows": 37,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 13,
                "maximum_rows": 18,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 10,
                "maximum_rows": 13,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 7,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 4,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "compact_async_positive_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_compact_async_positive_f32.cu",
        "source_sha256": "59262d0c231ec43a89a0c03c2f994e5e205d3591b01eb45a8bed9133cc51bcd9",
        "kernel_symbol": "kernel_cake_fused_kda_decode_compact_async_positive_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 36480,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 25,
                "maximum_rows": 37,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 13,
                "maximum_rows": 18,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 10,
                "maximum_rows": 13,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 7,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 4,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "compact_async_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_compact_async_f32_wide_slot_offsets.cu",
        "source_sha256": "ae12e12cd9fbad7a6a182f9803427e26cd60d69ff4c10bc95cdc47c636c42ed7",
        "kernel_symbol": "kernel_cake_fused_kda_decode_compact_async_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 36480,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 25,
                "maximum_rows": 37,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 13,
                "maximum_rows": 18,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 10,
                "maximum_rows": 13,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 7,
                "maximum_rows": 9,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 4,
                "maximum_rows": 4,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "compact_async_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_compact_async_f32.cu",
        "source_sha256": "9469b9e49f35d4b9880e1177c37b1a7210f554c936371a0f2bc48da2e98b8236",
        "kernel_symbol": "kernel_cake_fused_kda_decode_compact_async_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 36480,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 25,
                "maximum_rows": 37,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 13,
                "maximum_rows": 18,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 10,
                "maximum_rows": 13,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 7,
                "maximum_rows": 9,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 4,
                "maximum_rows": 4,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "compact_async_bf16_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_compact_async_bf16_wide_slot_offsets.cu",
        "source_sha256": "9f8aca911be34bb77b5e8c03ec104129c9150d9b27e052d6ad36bfaa028b3452",
        "kernel_symbol": "kernel_cake_fused_kda_decode_compact_async_bf16_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 36480,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 25,
                "maximum_rows": 37,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 13,
                "maximum_rows": 18,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 10,
                "maximum_rows": 13,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 7,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 4,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "compact_async_bf16",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_compact_async_bf16.cu",
        "source_sha256": "3613d50fc122fb1a6fc7fded96c52243b968d1b4a3f74ae1a5b4d6a1dd50159a",
        "kernel_symbol": "kernel_cake_fused_kda_decode_compact_async_bf16",
        "abi_kind": "standard",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 36480,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 25,
                "maximum_rows": 37,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 13,
                "maximum_rows": 18,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 10,
                "maximum_rows": 13,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 7,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 4,
                "maximum_rows": 4,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_h96_pr_strides_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_h96_pr_strides_f32_wide_slot_offsets.cu",
        "source_sha256": "c1a6d8eb30fe4867e08fc423d4dd9ca5e884269d7ca044c5dd260ebf30b4daec",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_h96_pr_strides_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [96],
                "minimum_rows": 13,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": 36881,
                    "conv_slot_stride": 3256320,
                    "beta_row_stride": 97,
                    "state_slot_stride": 1628160,
                    "output_gate_row_stride": 12295,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_h96_pr_strides_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_h96_pr_strides_f32.cu",
        "source_sha256": "45cafd3027883fffd56e7fa599083ade2d1f123e776bace966f10e07dcb03e35",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_h96_pr_strides_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [96],
                "minimum_rows": 13,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": 36881,
                    "conv_slot_stride": 3256320,
                    "beta_row_stride": 97,
                    "state_slot_stride": 1628160,
                    "output_gate_row_stride": 12295,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_h96_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_h96_f32_wide_slot_offsets.cu",
        "source_sha256": "edb3376499e1804a50b076f368e08d8536a6527a3885da3e57d96cf618e7d384",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_h96_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [96],
                "minimum_rows": 13,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_h96_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_h96_f32.cu",
        "source_sha256": "977e6fc9d71e782b8a4e22652cde2979ddbe60b87c515d0172fd21ea5a5b89f9",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_h96_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [96],
                "minimum_rows": 13,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_pr_eval_h12_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_pr_eval_h12_f32_wide_slot_offsets.cu",
        "source_sha256": "32b32d0d6f85ea6c5a3f45810e03222f4def1802581bb50bd806310fba495145",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_pr_eval_h12_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 99,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 4625,
                    "conv_slot_stride": 407040,
                    "beta_row_stride": 13,
                    "state_slot_stride": 203520,
                    "output_gate_row_stride": 1543,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_pr_eval_h12_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_pr_eval_h12_f32.cu",
        "source_sha256": "f54cd47fd75958c011ed229d7c3950e433f442d3a826fbde556a39223971e118",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_pr_eval_h12_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 99,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 4625,
                    "conv_slot_stride": 407040,
                    "beta_row_stride": 13,
                    "state_slot_stride": 203520,
                    "output_gate_row_stride": 1543,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_pr_eval_h24_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_pr_eval_h24_f32_wide_slot_offsets.cu",
        "source_sha256": "826e650226ab395d89655bdef48888da0c84ca08cf4be09db7365f92856b91ee",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_pr_eval_h24_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [24],
                "minimum_rows": 50,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 9233,
                    "conv_slot_stride": 814080,
                    "beta_row_stride": 25,
                    "state_slot_stride": 407040,
                    "output_gate_row_stride": 3079,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_pr_eval_h24_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_pr_eval_h24_f32.cu",
        "source_sha256": "e239d1e8df0f36e3377fa4894de0e83cc611ee7e6e612dd73fe41fa403e73550",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_pr_eval_h24_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [24],
                "minimum_rows": 50,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 9233,
                    "conv_slot_stride": 814080,
                    "beta_row_stride": 25,
                    "state_slot_stride": 407040,
                    "output_gate_row_stride": 3079,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_pr_eval_h32_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_pr_eval_h32_f32_wide_slot_offsets.cu",
        "source_sha256": "35fe4e110f3b681dcb6a9351bfd0945992ed2cd0cb11e9ab4c2e4cceb86c8885",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_pr_eval_h32_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [32],
                "minimum_rows": 32,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 12305,
                    "conv_slot_stride": 1085440,
                    "beta_row_stride": 33,
                    "state_slot_stride": 542720,
                    "output_gate_row_stride": 4103,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_pr_eval_h32_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_pr_eval_h32_f32.cu",
        "source_sha256": "94789a2e54704efb8f33fdb5ffd854872dd3d8025051fcde24198fc0a78ec92d",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_pr_eval_h32_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [32],
                "minimum_rows": 32,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 12305,
                    "conv_slot_stride": 1085440,
                    "beta_row_stride": 33,
                    "state_slot_stride": 542720,
                    "output_gate_row_stride": 4103,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_pr_eval_h48_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_pr_eval_h48_f32_wide_slot_offsets.cu",
        "source_sha256": "46063a818a33bdcc5434ce362b7bb80b3b432d99192c37623e1350057a6d7c35",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_pr_eval_h48_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [48],
                "minimum_rows": 25,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 18449,
                    "conv_slot_stride": 1628160,
                    "beta_row_stride": 49,
                    "state_slot_stride": 814080,
                    "output_gate_row_stride": 6151,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_pr_eval_h48_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_pr_eval_h48_f32.cu",
        "source_sha256": "a193d02e2ef8af05f2fc76ed0ac0bda08a94a8859f076a4012a6f4ca8240c0d3",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_pr_eval_h48_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [48],
                "minimum_rows": 25,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 18449,
                    "conv_slot_stride": 1628160,
                    "beta_row_stride": 49,
                    "state_slot_stride": 814080,
                    "output_gate_row_stride": 6151,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_f32_wide_slot_offsets.cu",
        "source_sha256": "aed5432b253875af6ad224090bbc550a90385898dadba4bd99ca325d0025da9f",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 99,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 50,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 32,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 25,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "high_work_positive_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_positive_f32.cu",
        "source_sha256": "7b2cd480d53256a036fb4a97fc19d3559e01c1b97a4475ccb54f9f3eba22d65b",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_positive_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 99,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 50,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 32,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 25,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "high_work_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_f32_wide_slot_offsets.cu",
        "source_sha256": "b26155c4d8ce7287aa76df21628b2383980a96eb56a58f6a42671a0c2a2b1e37",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 99,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 50,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 32,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 25,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 13,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "high_work_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_f32.cu",
        "source_sha256": "95f6ab605553a70e3dfc546e98dccff87b4fe212162a09d3747c338c8abf15dd",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 28288,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 99,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 50,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 32,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 25,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 13,
                "maximum_rows": None,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "high_work_bf16_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_bf16_wide_slot_offsets.cu",
        "source_sha256": "d9ad9af6deb58f2b5e9a0bd254ff3325af80b85f803d8c09a02d85c2b7cba415",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_bf16_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 20096,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 99,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 50,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 32,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 25,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 13,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "high_work_bf16",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_high_work_bf16.cu",
        "source_sha256": "51549f43cf46630fe89c184664dd9ce6cee020c04a086d98acca2864b961233f",
        "kernel_symbol": "kernel_cake_fused_kda_decode_high_work_bf16",
        "abi_kind": "standard",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 20096,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 99,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 50,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 32,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 25,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 13,
                "maximum_rows": None,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "pr_eval_h32_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_pr_eval_h32_f32_wide_slot_offsets.cu",
        "source_sha256": "92db247ce7478f20facf934242d2bc41201c78438197ea471a2165c74032aaec",
        "kernel_symbol": "kernel_cake_fused_kda_decode_pr_eval_h32_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [32],
                "minimum_rows": 5,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 12305,
                    "conv_slot_stride": 1085440,
                    "beta_row_stride": 33,
                    "state_slot_stride": 542720,
                    "output_gate_row_stride": 4103,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 14,
                "maximum_rows": 31,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 12305,
                    "conv_slot_stride": 1085440,
                    "beta_row_stride": 33,
                    "state_slot_stride": 542720,
                    "output_gate_row_stride": 4103,
                },
            },
        ),
    },
    {
        "name": "pr_eval_h32_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_pr_eval_h32_f32.cu",
        "source_sha256": "b326da85485c059c5f5b0d615f2e0f56b5c9ddb19d8fb46ea5f5759e01171640",
        "kernel_symbol": "kernel_cake_fused_kda_decode_pr_eval_h32_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [32],
                "minimum_rows": 5,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 12305,
                    "conv_slot_stride": 1085440,
                    "beta_row_stride": 33,
                    "state_slot_stride": 542720,
                    "output_gate_row_stride": 4103,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 14,
                "maximum_rows": 31,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": [-5.0],
                "norm_eps_values": [1e-05],
                "strides": {
                    "x_row_stride": 12305,
                    "conv_slot_stride": 1085440,
                    "beta_row_stride": 33,
                    "state_slot_stride": 542720,
                    "output_gate_row_stride": 4103,
                },
            },
        ),
    },
    {
        "name": "direct_positive_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_direct_positive_f32_wide_slot_offsets.cu",
        "source_sha256": "7bd68ab1bb50d1893a8286607952a1665d6c2deb0835dfb08ad60fdeb87664a9",
        "kernel_symbol": "kernel_cake_fused_kda_decode_direct_positive_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 13,
                "maximum_rows": 24,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [12],
                "minimum_rows": 38,
                "maximum_rows": 98,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 7,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 5,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 14,
                "maximum_rows": 31,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 4,
                "maximum_rows": 6,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 10,
                "maximum_rows": 24,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 2,
                "maximum_rows": 3,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "direct_positive_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_direct_positive_f32.cu",
        "source_sha256": "b36599d0ee29016185ebc8c8aadd28589aa567192accac8d75947525fee8bc88",
        "kernel_symbol": "kernel_cake_fused_kda_decode_direct_positive_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 13,
                "maximum_rows": 24,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [12],
                "minimum_rows": 38,
                "maximum_rows": 98,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 7,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 5,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 14,
                "maximum_rows": 31,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 4,
                "maximum_rows": 6,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 10,
                "maximum_rows": 24,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 2,
                "maximum_rows": 3,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "direct_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_direct_f32_wide_slot_offsets.cu",
        "source_sha256": "266d7c402696773f10b44b795807ff74848e894fdc7e02e85bd566539824898e",
        "kernel_symbol": "kernel_cake_fused_kda_decode_direct_f32_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 13,
                "maximum_rows": 24,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [12],
                "minimum_rows": 38,
                "maximum_rows": 98,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 7,
                "maximum_rows": 12,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 19,
                "maximum_rows": 49,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 5,
                "maximum_rows": 9,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 14,
                "maximum_rows": 31,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 4,
                "maximum_rows": 6,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 10,
                "maximum_rows": 24,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 2,
                "maximum_rows": 3,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 5,
                "maximum_rows": 12,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 19,
                "maximum_rows": 49,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 5,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "direct_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_direct_f32.cu",
        "source_sha256": "1d0bfaab93c0004f34c4b48d09cd3cdd18496dcd90b81e7aa4c11d4c5a2f13f1",
        "kernel_symbol": "kernel_cake_fused_kda_decode_direct_f32",
        "abi_kind": "standard",
        "state_dtype": "float32",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 13,
                "maximum_rows": 24,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [12],
                "minimum_rows": 38,
                "maximum_rows": 98,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 7,
                "maximum_rows": 12,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 19,
                "maximum_rows": 49,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 5,
                "maximum_rows": 9,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 14,
                "maximum_rows": 31,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 4,
                "maximum_rows": 6,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 10,
                "maximum_rows": 24,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 2,
                "maximum_rows": 3,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 5,
                "maximum_rows": 12,
                "state_indices_modes": ["unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 19,
                "maximum_rows": 49,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 5,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "direct_bf16_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_direct_bf16_wide_slot_offsets.cu",
        "source_sha256": "e690b987ec10f65644ed434a3ee5f12575b553c810a0faf811812805aed7e3e7",
        "kernel_symbol": "kernel_cake_fused_kda_decode_direct_bf16_wide_slot_offsets",
        "abi_kind": "standard",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 64,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 13,
                "maximum_rows": 24,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [12],
                "minimum_rows": 38,
                "maximum_rows": 98,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 7,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 19,
                "maximum_rows": 49,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 5,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 14,
                "maximum_rows": 31,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 4,
                "maximum_rows": 6,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 10,
                "maximum_rows": 24,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 2,
                "maximum_rows": 3,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 5,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
    {
        "name": "direct_bf16",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_direct_bf16.cu",
        "source_sha256": "c1f0b9901615a9d41705fd25b77559e8c869e1fa17df645198b37450afe67b98",
        "kernel_symbol": "kernel_cake_fused_kda_decode_direct_bf16",
        "abi_kind": "standard",
        "state_dtype": "bfloat16",
        "slot_offset_bits": 32,
        "extra_cuda_cflags": ("--use_fast_math",),
        "threads": 256,
        "dynamic_smem_bytes": 3712,
        "eligibility": (
            {
                "heads": [12],
                "minimum_rows": 13,
                "maximum_rows": 24,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [12],
                "minimum_rows": 38,
                "maximum_rows": 98,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 7,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [24],
                "minimum_rows": 19,
                "maximum_rows": 49,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 5,
                "maximum_rows": 9,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [32],
                "minimum_rows": 14,
                "maximum_rows": 31,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 4,
                "maximum_rows": 6,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [48],
                "minimum_rows": 10,
                "maximum_rows": 24,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 2,
                "maximum_rows": 3,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
            {
                "heads": [96],
                "minimum_rows": 5,
                "maximum_rows": 12,
                "state_indices_modes": ["positive_unique", "unique_or_null"],
                "lower_bound_values": "any",
                "norm_eps_values": "any",
                "strides": {
                    "x_row_stride": None,
                    "conv_slot_stride": None,
                    "beta_row_stride": None,
                    "state_slot_stride": None,
                    "output_gate_row_stride": None,
                },
            },
        ),
    },
)


def _values_or_any(value: object) -> tuple[Any, ...] | None:
    if value == "any":
        return None
    if not isinstance(value, list):
        raise ValueError("Cake fused KDA eligibility values must be a list or 'any'")
    return tuple(value)


@functools.cache
def get_cake_fused_kda_decode_variants() -> tuple[CakeFusedKDADecodeVariant, ...]:
    """Resolve and verify the explicitly registered checked-in device sources."""

    csrc_dir = get_kda_csrc_dir().resolve()
    result: list[CakeFusedKDADecodeVariant] = []
    observed: set[tuple[str, str]] = set()
    for item in _VARIANT_SPECS:
        name = item["name"]
        target = item["target"]
        key = (name, target)
        if key in observed:
            raise ValueError(f"duplicate Cake fused KDA variant {name}/{target}")
        observed.add(key)
        body = (csrc_dir / item["body"]).resolve()
        if body.parent != csrc_dir or not body.name.startswith(
            "cake_fused_kda_decode_"
        ):
            raise ValueError(f"invalid Cake fused KDA source path: {body}")
        if not body.is_file():
            raise FileNotFoundError(f"Cake fused KDA source not found: {body}")
        source_sha256 = hashlib.sha256(body.read_bytes()).hexdigest()
        if source_sha256 != item["source_sha256"]:
            raise ValueError(
                f"Cake fused KDA source identity mismatch for {name}: "
                f"{source_sha256} != {item['source_sha256']}"
            )
        eligibility = []
        for rule in item["eligibility"]:
            eligibility.append(
                CakeFusedKDADecodeEligibility(
                    heads=tuple(rule["heads"]),
                    minimum_rows=rule["minimum_rows"],
                    maximum_rows=rule["maximum_rows"],
                    state_indices_modes=tuple(rule["state_indices_modes"]),
                    lower_bound_values=_values_or_any(rule["lower_bound_values"]),
                    norm_eps_values=_values_or_any(rule["norm_eps_values"]),
                    strides=tuple(rule["strides"].items()),
                )
            )
        result.append(
            CakeFusedKDADecodeVariant(
                name=name,
                target=target,
                body_path=body,
                source_sha256=source_sha256,
                kernel_symbol=item["kernel_symbol"],
                abi_kind=item["abi_kind"],
                state_dtype=item["state_dtype"],
                slot_offset_bits=item["slot_offset_bits"],
                extra_cuda_cflags=tuple(item["extra_cuda_cflags"]),
                threads=item["threads"],
                dynamic_smem_bytes=item["dynamic_smem_bytes"],
                eligibility=tuple(eligibility),
            )
        )
    return tuple(result)


def _variant_build_identity_payload(
    variant: CakeFusedKDADecodeVariant,
) -> dict[str, Any]:
    binding_header = get_kda_csrc_dir().resolve() / _BINDING_HEADER
    if not binding_header.is_file():
        raise FileNotFoundError(f"Cake fused KDA binding not found: {binding_header}")
    return {
        "name": variant.name,
        "target": variant.target,
        "body": variant.body_path.name,
        "source_sha256": variant.source_sha256,
        "kernel_symbol": variant.kernel_symbol,
        "abi_kind": variant.abi_kind,
        "abi": CAKE_FUSED_KDA_DECODE_ABIS[variant.abi_kind],
        "arg_plan_sha256": _ARG_PLAN_SHA256[variant.abi_kind],
        "state_dtype": variant.state_dtype,
        "slot_offset_bits": variant.slot_offset_bits,
        "target_nvcc_flags": sm100a_nvcc_flags,
        "target_define": _TARGET_DEFINE,
        "extra_cuda_cflags": variant.extra_cuda_cflags,
        "threads": variant.threads,
        "dynamic_smem_bytes": variant.dynamic_smem_bytes,
        "eligibility": [
            {
                "heads": rule.heads,
                "minimum_rows": rule.minimum_rows,
                "maximum_rows": rule.maximum_rows,
                "state_indices_modes": rule.state_indices_modes,
                "lower_bound_values": rule.lower_bound_values,
                "norm_eps_values": rule.norm_eps_values,
                "strides": rule.strides,
            }
            for rule in variant.eligibility
        ],
        "binding_header_sha256": hashlib.sha256(
            binding_header.read_bytes()
        ).hexdigest(),
        "rendered_binding_sha256": hashlib.sha256(
            _render_binding(variant).encode()
        ).hexdigest(),
    }


def _variant_build_identity(variant: CakeFusedKDADecodeVariant) -> str:
    return hashlib.sha256(
        json.dumps(
            _variant_build_identity_payload(variant),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def get_cake_fused_kda_decode_program_identity() -> str:
    """Return a stable identity for the registered executable closure."""

    variants = get_cake_fused_kda_decode_variants()
    payload = {
        "schema": "cake.fused_kda_decode.program.v1",
        "variants": [_variant_build_identity_payload(variant) for variant in variants],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _eligibility_matches(
    rule: CakeFusedKDADecodeEligibility,
    *,
    num_heads: int,
    num_rows: int,
    state_indices_mode: CakeFusedKDADecodeStateIndicesMode,
    lower_bound: float | None,
    norm_eps: float,
    strides: dict[str, int],
) -> bool:
    if num_heads not in rule.heads or num_rows < rule.minimum_rows:
        return False
    if rule.maximum_rows is not None and num_rows > rule.maximum_rows:
        return False
    if state_indices_mode not in rule.state_indices_modes:
        return False
    if (
        rule.lower_bound_values is not None
        and lower_bound not in rule.lower_bound_values
    ):
        return False
    if rule.norm_eps_values is not None and norm_eps not in rule.norm_eps_values:
        return False
    return all(
        expected is None or strides[name] == expected for name, expected in rule.strides
    )


def select_cake_fused_kda_decode_variant(
    *,
    target: CakeFusedKDADecodeTarget,
    num_heads: int,
    num_rows: int,
    num_slots: int,
    state_dtype: str,
    state_indices_mode: CakeFusedKDADecodeStateIndicesMode,
    lower_bound: float | None,
    norm_eps: float,
    x_row_stride: int,
    conv_slot_stride: int,
    beta_row_stride: int,
    state_slot_stride: int,
    output_gate_row_stride: int,
    variants: Sequence[CakeFusedKDADecodeVariant] | None = None,
) -> CakeFusedKDADecodeVariant | None:
    """Select one Cake route using host scalars and tensor metadata only."""

    if target not in _TARGETS:
        raise ValueError(f"unsupported Cake fused KDA target: {target}")
    if num_heads not in (12, 24, 32, 48, 96):
        raise ValueError(f"unsupported Cake fused KDA head count: {num_heads}")
    if num_rows <= 0 or num_slots <= 0:
        raise ValueError("Cake fused KDA rows and slots must be positive")
    if state_dtype not in ("bfloat16", "float32"):
        raise ValueError(f"unsupported Cake fused KDA state dtype: {state_dtype}")
    if state_indices_mode not in _STATE_INDICES_MODES:
        raise ValueError(
            f"unsupported Cake fused KDA state_indices_mode: {state_indices_mode}"
        )
    if lower_bound is not None and (
        not math.isfinite(float(lower_bound)) or float(lower_bound) >= 0.0
    ):
        raise ValueError("lower_bound must be finite, negative, or None")
    if not math.isfinite(float(norm_eps)) or float(norm_eps) < 0.0:
        raise ValueError("norm_eps must be finite and non-negative")
    strides = {
        "x_row_stride": x_row_stride,
        "conv_slot_stride": conv_slot_stride,
        "beta_row_stride": beta_row_stride,
        "state_slot_stride": state_slot_stride,
        "output_gate_row_stride": output_gate_row_stride,
    }
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in strides.values()
    ):
        raise ValueError("Cake fused KDA strides must be non-negative integers")
    if num_rows > 2**31 - 1 or any(value > 2**31 - 1 for value in strides.values()):
        return None
    qkv_size = 3 * num_heads * 128
    required_slot_offset_bits = (
        64
        if (
            (num_slots - 1) * conv_slot_stride + 3 * qkv_size - 1 > 2**31 - 1
            or (num_slots - 1) * state_slot_stride + num_heads * 128 * 128 - 1
            > 2**31 - 1
        )
        else 32
    )
    available = (
        get_cake_fused_kda_decode_variants() if variants is None else tuple(variants)
    )
    for variant in available:
        if (
            variant.target != target
            or variant.state_dtype != state_dtype
            or variant.slot_offset_bits != required_slot_offset_bits
        ):
            continue
        if any(
            _eligibility_matches(
                rule,
                num_heads=num_heads,
                num_rows=num_rows,
                state_indices_mode=state_indices_mode,
                lower_bound=lower_bound,
                norm_eps=norm_eps,
                strides=strides,
            )
            for rule in variant.eligibility
        ):
            return variant
    return None


def cake_fused_kda_decode_is_available() -> bool:
    """Return whether the explicitly registered Cake source closure is installed."""

    return bool(get_cake_fused_kda_decode_variants())


def get_cake_fused_kda_decode_variant(
    name: str, target: CakeFusedKDADecodeTarget
) -> CakeFusedKDADecodeVariant:
    for variant in get_cake_fused_kda_decode_variants():
        if variant.name == name and variant.target == target:
            return variant
    raise RuntimeError(f"Cake fused KDA source is unavailable for {name}/{target}")


def get_cake_fused_kda_decode_uri(name: str, target: CakeFusedKDADecodeTarget) -> str:
    variant = get_cake_fused_kda_decode_variant(name, target)
    return (
        f"cake_fused_kda_decode_{name}_{_variant_build_identity(variant)[:16]}_{target}"
    )


def _kernel_declaration(variant: CakeFusedKDADecodeVariant) -> str:
    state_type = "__nv_bfloat16*" if variant.state_dtype == "bfloat16" else "float*"
    arguments = [
        "__nv_bfloat16* x",
        "float* weight",
        "__nv_bfloat16* conv_state",
        "__nv_bfloat16* raw_gate",
        "__nv_bfloat16* raw_beta",
        "float* A_log",
        "float* dt_bias",
        "int* state_indices",
        f"{state_type} state",
        "__nv_bfloat16* output_gate",
        "float* norm_weight",
        "__nv_bfloat16* output",
        "int x_row_stride",
        "int conv_slot_stride",
        "int beta_row_stride",
        "int state_slot_stride",
        "int output_gate_row_stride",
        "int H",
    ]
    if variant.abi_kind == "repeated_safe":
        arguments.append("int rows")
    arguments.extend(
        ["int use_lower_bound", "float lower_bound_log2", "float norm_eps"]
    )
    return (
        f'extern "C" __global__ void {variant.kernel_symbol}(\n    '
        + ",\n    ".join(arguments)
        + ");"
    )


def _render_binding(variant: CakeFusedKDADecodeVariant) -> str:
    has_rows = int(variant.abi_kind == "repeated_safe")
    state_is_bfloat16 = int(variant.state_dtype == "bfloat16")
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>

{_kernel_declaration(variant)}

#define FLASHINFER_CAKE_FUSED_KDA_DECODE_KERNEL {variant.kernel_symbol}
#define FLASHINFER_CAKE_FUSED_KDA_DECODE_THREADS {variant.threads}
#define FLASHINFER_CAKE_FUSED_KDA_DECODE_SMEM_BYTES {variant.dynamic_smem_bytes}
#define FLASHINFER_CAKE_FUSED_KDA_DECODE_HAS_ROWS {has_rows}
#define FLASHINFER_CAKE_FUSED_KDA_DECODE_STATE_IS_BFLOAT16 {state_is_bfloat16}
#define FLASHINFER_CAKE_FUSED_KDA_DECODE_ARG_PLAN_SHA256 "{_ARG_PLAN_SHA256[variant.abi_kind]}"

#include "{_BINDING_HEADER}"
"""


@functools.cache
def gen_cake_fused_kda_decode_module(
    name: str, target: CakeFusedKDADecodeTarget
) -> JitSpec:
    """Build a Cake module from separate device and binding translation units."""

    variant = get_cake_fused_kda_decode_variant(name, target)
    csrc_dir = get_kda_csrc_dir()
    binding_header = csrc_dir / _BINDING_HEADER
    if not binding_header.is_file():
        raise FileNotFoundError(f"Cake fused KDA binding not found: {binding_header}")
    uri = get_cake_fused_kda_decode_uri(name, target)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_fused_kda_decode_binding.cu"
    write_if_different(binding, _render_binding(variant))
    spec = gen_kda_jit_spec(
        name=uri,
        sources=[variant.body_path, binding],
        target=target,
        target_define=_TARGET_DEFINE,
        csrc_dir=csrc_dir,
        include_dir=get_flashinfer_include_dir(),
        extra_cuda_cflags=variant.extra_cuda_cflags,
    )
    logger.info(
        f"Generated Cake fused KDA decode {name} {target} JIT spec: {spec.name}"
    )
    return spec


@functools.cache
def load_cake_fused_kda_decode_module(name: str, target: CakeFusedKDADecodeTarget):
    module = gen_cake_fused_kda_decode_module(name, target).build_and_load()
    logger.info(f"Loaded Cake fused KDA decode {name} {target} module")
    return module


__all__ = [
    "CAKE_FUSED_KDA_DECODE_ABIS",
    "CakeFusedKDADecodeEligibility",
    "CakeFusedKDADecodeStateIndicesMode",
    "CakeFusedKDADecodeTarget",
    "CakeFusedKDADecodeVariant",
    "cake_fused_kda_decode_is_available",
    "gen_cake_fused_kda_decode_module",
    "get_cake_fused_kda_decode_program_identity",
    "get_cake_fused_kda_decode_uri",
    "get_cake_fused_kda_decode_variant",
    "get_cake_fused_kda_decode_variants",
    "load_cake_fused_kda_decode_module",
    "select_cake_fused_kda_decode_variant",
]
