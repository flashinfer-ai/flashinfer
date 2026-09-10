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
from .core import JitSpec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags
from .utils import write_if_different

CakeFusedKDADecodeTarget = Literal["sm100a", "sm103a"]
CakeFusedKDADecodeStateIndicesMode = Literal[
    "positive_unique",
    "unique_or_null",
    "repeated_positive",
]

_BINDING_HEADER = "cake_fused_kda_decode_binding.cuh"
_TARGETS: tuple[CakeFusedKDADecodeTarget, ...] = ("sm100a", "sm103a")
_STATE_INDICES_MODES: tuple[CakeFusedKDADecodeStateIndicesMode, ...] = (
    "positive_unique",
    "unique_or_null",
    "repeated_positive",
)
_TARGET_DEFINES = {
    "sm100a": "-DFLASHINFER_CAKE_FUSED_KDA_DECODE_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_CAKE_FUSED_KDA_DECODE_TARGET_MINOR=3",
}
_TARGET_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

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
    """One explicitly registered CUDA source and launch description."""

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
# The SM100a source and launch records are shared with explicit SM103a builds;
# compilation flags and executable identities remain target-specific.
_VARIANT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "repeated_safe_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_repeated_safe_f32_wide_slot_offsets.cu",
        "source_sha256": "21e73e321ae646ecd42e4e0371dcbf221181c232a86256cb70a3ccd865dc61a1",
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
        "source_sha256": "bebf6ca51c60762f25109a3d04a4252799fa73dc44bae1241a122bca6d198344",
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
        "source_sha256": "1c174acdf2e0248e60f3a511fcfa9a50824ee9d03d570cd03396eb6ba5838ce4",
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
        "source_sha256": "dc158ec5ba21324b1d56253728061ea0a11a470ab63c216cad870b5824449969",
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
        "source_sha256": "30c935b217f9b0816ae08e39670bc9eeaefa8236fbd6d0939da0483ab5583a81",
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
        "source_sha256": "8007333b792d6326013fbbad4b5e03d43a1102e0679e5163632228e7caf32813",
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
        "source_sha256": "09a41e70a3dd5348677850bd8368c250b6b50fb9e06a610775cb3b944265f246",
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
        "source_sha256": "3b21bcd92074be17ba227f92e1c8ce5bf56b7f43f854d4923075132e799b0809",
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
        "source_sha256": "44cba2d86048fbd8aa86cd6cd3982bc49462de369aeca000650e24a652005aa0",
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
        "source_sha256": "e359123947fdeb518197fe4b4c8dbe2693721b67992c318e386c70caeedea18f",
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
        "source_sha256": "4c4135c88735554cbad1a7705c38dead8a1e0e3168d088902ae09b1f89b0880b",
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
        "source_sha256": "00d0f7fe2de2da8257f44264f0a32e42b6be7a4df7bebaa2d89756cfa5f9bd91",
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
        "source_sha256": "0a7b86dd9a60afbf84afa21f5ecfb31ec89150ae9a22538cf032295527636f3e",
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
        "source_sha256": "44b56bafee24337343e5e44f9dc00e01da957ec1dd3b510994bd1fac69c5214d",
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
        "source_sha256": "2ce6119617b8e79e87fa57bb7bbc3966011679813f792cc6b418bb1208a21a82",
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
        "source_sha256": "3e41686d9b192afc9b398afb40195148267e7de16fa37ed668a38e9b777945e5",
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
        "source_sha256": "0598ef9796354443c4fcd53d9d71da7a1fcee0293e17b0fa59bfdae07862a3c6",
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
        "source_sha256": "3adc41699989811b046e6bc69cea1037e3ebf860d192baf737271a1ccd833396",
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
        "source_sha256": "ba50f5b01c4fcd94f9328f214a48be346862c9b5835b5a49fee49a007c83d5b2",
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
        "source_sha256": "de506948c7461ccdd3b9096ba1829ab24bfe1340089db17ff2fe316c8cefadb5",
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
        "source_sha256": "dcca3d4a4026df573e050e89ed6610088d7a0fbffd94ef76a1bbb0d66d0c764a",
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
        "source_sha256": "645abf5a9a52088e7847094b85f8bfbde37e5d51f0a5f7b030efb19408f42bcf",
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
        "source_sha256": "5f9dfd3373bae33b16d14df09886211aee482f3c2915a350f0999e2d1a93acb8",
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
        "source_sha256": "f9bc92881e276fb121c23dfa738f5b10b9cbee03e3522af46dd121219179c8e3",
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
        "source_sha256": "dcf31d465d01498cad1fab9aece2a6b71e9a3e859b6b429265b57b9bc9b9b146",
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
        "source_sha256": "b915d7d86203f9ea3937249764fda157cfd7580875bbc9dc83194d6ebf2a772c",
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
        "source_sha256": "966f40f6befc85af15ef49e95c0e0bd2c378c0c33e3986fdbdae911d511d1fba",
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
        "source_sha256": "d93a99bb40a1f295ef8d7ea2a22ee6e76231e55d11f94bd873820202679bb3a6",
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
        "source_sha256": "61d06cab16f49f386592da65c5c75ba91f6c8bb484a5918f21a409abe729114a",
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
        "source_sha256": "e56ca462ec5242a47f4458e21ce3137b5970e53d5378f8b83843b4def351d9bf",
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
        "source_sha256": "08c5efcabdf3a4c572a001ad6f8a2bbb6f6655d4782a7c1b37d2240fe124005c",
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
        "source_sha256": "50b6938de6087d55e8a49a1b32565d9550bcf0434d878df65b5f872d3664d055",
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
        "source_sha256": "749089b461c5bd496020f18e0328eb470fc0d0f6a46d49971d3e057e7bec02dc",
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
        "source_sha256": "6c3f7ed5b393c82193f67f278a91c714ba9c31cd9910e8510c38b02df7723661",
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
        "source_sha256": "7e014e7123fc951de66536fc63ce67262e18132cd60b4df7e318451e1e039d0d",
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
        "source_sha256": "96cc54207ad165077ed8957cb73943e8cc2160f9d7e77953750beac534132a80",
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
        "source_sha256": "a0de3fe324a51466729e87807752cf7efd90f15e3e75c1109b8d32d3e2a3971c",
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
        "source_sha256": "b2e14eda831555f47833d025ec557d1aacbadab375354a49ba0e2bb6ab42594d",
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
        "source_sha256": "a1cb811e06eb40e9bbfe620c51d8c6aadf58995adc0a4667e5050ab133925908",
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
        "source_sha256": "a1780b648ce3b8796dca741fce5a1c5d31c869ba3ef20252a247a9bdc6d3a7ab",
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
        "source_sha256": "7555e58deb93585d0465138fe9fc039bc69d05d8851923b26ff85f4fb01fec5f",
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
        "source_sha256": "b3036d0419527b0be37a93f85c2ecfb4010be366d2050a26d59aa973164456f6",
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
        "source_sha256": "5807b7a14c16753a0188e2f05f7e072d9d6e11dd3466f53124bfec9091fd3514",
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
        "source_sha256": "3019114b08e2586059575fab558acdc9c023f64198676ba88fd2d40e12670973",
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
    {
        "name": "wide512_vector4_positive_f32",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_wide512_vector4_positive_f32.cu",
        "source_sha256": "b2f7be76d44fe4a2619b60c271bd08ba7c7fd930aec7509d6606bf8cd1ac9a30",
        "kernel_symbol": "kernel_cake_fused_kda_decode_wide512_vector4_positive_f32",
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
                "minimum_rows": 1,
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
        "name": "wide512_vector4_positive_f32_wide_slot_offsets",
        "target": "sm100a",
        "body": "cake_fused_kda_decode_wide512_vector4_positive_f32_wide_slot_offsets.cu",
        "source_sha256": "dabef8ba748408cd154e802db322d82132040065bd6cd16753c11c5e10fad03d",
        "kernel_symbol": "kernel_cake_fused_kda_decode_wide512_vector4_positive_f32_wide_slot_offsets",
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
                "minimum_rows": 1,
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
)


def _values_or_any(value: object) -> tuple[Any, ...] | None:
    if value == "any":
        return None
    if not isinstance(value, list):
        raise ValueError("Cake fused KDA eligibility values must be a list or 'any'")
    return tuple(value)


@functools.cache
def get_cake_fused_kda_decode_variants(
    target: CakeFusedKDADecodeTarget = "sm100a",
) -> tuple[CakeFusedKDADecodeVariant, ...]:
    """Resolve the shared source registry for one explicit compilation target."""

    if target not in _TARGETS:
        raise ValueError(f"unsupported Cake fused KDA target: {target}")
    csrc_dir = get_kda_csrc_dir().resolve()
    result: list[CakeFusedKDADecodeVariant] = []
    observed: set[tuple[str, str]] = set()
    for item in _VARIANT_SPECS:
        name = item["name"]
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
        extra_cuda_cflags = tuple(item["extra_cuda_cflags"])
        if name in (
            "compact_async_f32_wide_slot_offsets",
            "compact_async_pr_eval_h96_f32_wide_slot_offsets",
        ):
            # Preserve three resident CTAs when NVCC allocates more registers
            # than the source compiler for these wide-offset schedules.
            extra_cuda_cflags += ("-Xptxas=--minnctapersm=3",)
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
                extra_cuda_cflags=extra_cuda_cflags,
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
        "target_nvcc_flags": _TARGET_NVCC_FLAGS[variant.target],
        "target_define": _TARGET_DEFINES[variant.target],
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


def get_cake_fused_kda_decode_program_identity(
    target: CakeFusedKDADecodeTarget = "sm100a",
) -> str:
    """Return a stable identity for one target's registered executable closure."""

    variants = get_cake_fused_kda_decode_variants(target)
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
    check_row_bounds: bool = True,
) -> bool:
    if num_heads not in rule.heads:
        return False
    if check_row_bounds and (
        num_rows < rule.minimum_rows
        or (rule.maximum_rows is not None and num_rows > rule.maximum_rows)
    ):
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


def _positive_f32_variants(num_heads: int, num_rows: int) -> tuple[str, ...]:
    """Choose positive-unique FP32 schedules by resident-CTA wave capacity."""

    sm_count = 148
    work_items = num_heads * num_rows
    if 2 * sm_count < work_items <= 3 * sm_count:
        return ("compact_async_pr_eval_h96_f32", "compact_async_positive_f32")
    partial_wave = work_items % (2 * sm_count)
    high_work = (
        (num_heads == 32 and num_rows >= 32)
        or (work_items >= 8 * sm_count)
        or (
            work_items > 4 * sm_count
            and 0 < partial_wave <= sm_count // 2
            and (num_heads != 12 or partial_wave >= num_heads)
        )
    )
    if high_work:
        return (
            f"high_work_positive_pr_eval_h{num_heads}_f32",
            "high_work_positive_h96_pr_strides_f32",
            "high_work_positive_h96_f32",
            "high_work_positive_f32",
        )
    # Pair-channel producers win the measured H12/22 and H24/10 cases.
    if (
        num_heads <= 24
        and (num_heads, num_rows) not in ((12, 22), (24, 10))
        and 3 * sm_count < 2 * work_items <= 4 * sm_count
    ):
        return ("wide512_vector4_positive_f32",)
    return ("wide512_positive_f32",)


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
        get_cake_fused_kda_decode_variants(target)
        if variants is None
        else tuple(variants)
    )
    positive_f32_variants = (
        _positive_f32_variants(num_heads, num_rows)
        if state_dtype == "float32" and state_indices_mode == "positive_unique"
        else None
    )
    for variant in available:
        if (
            variant.target != target
            or variant.state_dtype != state_dtype
            or variant.slot_offset_bits != required_slot_offset_bits
        ):
            continue
        if (
            positive_f32_variants is not None
            and variant.name.removesuffix("_wide_slot_offsets")
            not in positive_f32_variants
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
                check_row_bounds=positive_f32_variants is None,
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
    for variant in get_cake_fused_kda_decode_variants(target):
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
        target_define=_TARGET_DEFINES[target],
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
