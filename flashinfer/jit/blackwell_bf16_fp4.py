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

# JIT loader for the standalone Blackwell BF16 x FP4 GEMM bundle.

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from tvm_ffi import cpp

from . import env as jit_env


_SOURCE_NAMES = {
    "sm100": "flashinfer_blackwell_bf16_fp4_generated_sm100.cu",
    "sm103": "flashinfer_blackwell_bf16_fp4_generated_sm103.cu",
}
_MANIFEST_NAMES = {
    "sm100": "flashinfer_blackwell_bf16_fp4_generated_sm100.abi.json",
    "sm103": "flashinfer_blackwell_bf16_fp4_generated_sm103.abi.json",
}
_NVCC_ARCH = {"sm100": "sm_100a", "sm103": "sm_103a"}
_TARGET_SM = {"sm100": 100, "sm103": 103}
_TARGET_MINOR = {"sm100": 0, "sm103": 3}
_BINDING_NAME = "flashinfer_blackwell_bf16_fp4_binding.cu"

_COMMON_MANIFEST_KEYS = {
    "schema_version",
    "bundle",
    "arch",
    "tma_abi",
    "tensor_map_abi",
    "adapter_boundary",
}
_INTEGRATION_MANIFEST_KEYS = _COMMON_MANIFEST_KEYS | {
    "prepared_abis",
    "ir_symbols",
    "kernels",
    "dispatch",
}
_VARIANT_MANIFEST_KEYS = _COMMON_MANIFEST_KEYS | {
    "variants",
    "dispatcher",
    "composite_routes",
    "workspaces",
}
_TENSOR_MAP_ABI = {
    "public_type": "FlashInferTensorMap",
    "cuda_type": "CUtensorMap",
    "size_bytes": 128,
    "alignment_bytes": 128,
}
_PREPARED_ABIS = {
    "cudnn": {
        "B": {"dtype": "uint8", "shape": ["N", "K/2"]},
        "B_descale": {"dtype": "float8_e4m3fn", "shape": ["N", "K/16"]},
    },
    "cute_dsl": {
        "B": {"dtype": "int32", "shape": ["K/16", "N*2"]},
        "B_descale": {"dtype": "uint8", "shape": ["K/16", "N"]},
    },
}
_DISPATCH_INPUTS = [
    "backend",
    "out_dtype",
    "M",
    "N",
    "K",
    "has_alpha",
    "enable_pdl",
]
_DISPATCH_SELECTION = [
    {
        "components": ["cudnn_group_m128_bf16"],
        "route": "prepared_native_tcgen05_cudnn_e4m3_group_m128_v1",
        "when": {
            "backend": "cudnn",
            "out_dtype": "bfloat16",
            "prepared_transport": "tma",
            "shape": [768, 2112, 2048],
        },
    },
    {
        "components": [
            "cudnn_split_k2_partial_f32",
            "cudnn_split_k2_reduce_bf16",
        ],
        "route": "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "when": {
            "backend": "cudnn",
            "out_dtype": "bfloat16",
            "prepared_transport": "tma",
            "shape": [1, 4096, 4096],
        },
    },
    {
        "components": ["cute_warp_mma_m16_k16_bf16"],
        "route": (
            "prepared_native_warp_mma_m16n64k16_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {"K": 16, "backend": "cute-dsl"},
    },
    {
        "components": ["cute_warp_mma_m16_k32_bf16"],
        "route": (
            "prepared_native_warp_mma_m16n64k32_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {"K": 32, "backend": "cute-dsl"},
    },
    {
        "components": ["cute_warp_mma_m16_k48_bf16"],
        "route": (
            "prepared_native_warp_mma_m16n64k48_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {"K": 48, "backend": "cute-dsl"},
    },
    {
        "components": ["cute_warp_mma_m16_bf16"],
        "route": (
            "prepared_native_warp_mma_m16n64k128_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {
            "K_at_least": 128,
            "K_multiple": 64,
            "M_at_most": 16,
            "backend": "cute-dsl",
        },
    },
    {
        "components": ["cute_warp_mma_m32_bf16"],
        "route": (
            "prepared_native_warp_mma_m32n64k128_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {
            "K_at_least": 128,
            "K_multiple": 64,
            "M_between_inclusive": [17, 32],
            "backend": "cute-dsl",
        },
    },
    {
        "components": ["cute_warp_mma_m64_bf16"],
        "route": (
            "prepared_native_warp_mma_m64n64k128_cute_dsl_"
            "s0e5m3_f16mma_v2"
        ),
        "when": {
            "K_multiple": 128,
            "M_at_least": 33,
            "backend": "cute-dsl",
        },
    },
    {
        "components": ["cute_bf16"],
        "route": "prepared_native_tcgen05_cute_dsl_s0e5m3_v1",
        "when": {"backend": "cute-dsl", "fallback": True},
    },
    {
        "components": ["cudnn_cp_async_bf16", "cudnn_cp_async_f16"],
        "route": "prepared_native_tcgen05_cudnn_e4m3_cp_async_v1",
        "when": {"K_not_multiple": 256, "backend": "cudnn"},
    },
    {
        "components": ["cudnn_tma_bf16", "cudnn_tma_f16"],
        "route": "prepared_native_tcgen05_cudnn_e4m3_tma_v1",
        "when": {"K_multiple": 256, "backend": "cudnn"},
    },
]
_DISPATCHER = {
    "generic_grid_selection": {
        "flat_otherwise": True,
        "two_dimensional_when": {"ceil_div_M_16_at_most": 65535},
    },
    "selection_order": _DISPATCH_SELECTION,
    "selection_semantics": "first_match",
    "semantic_entrypoint": "flashinfer_blackwell_bf16_fp4_gemm",
    "variant_axes": ["output_dtype", "has_alpha", "enable_pdl", "grid_kind"],
}
_COMPOSITE_ROUTES = [
    {
        "launches": [
            {
                "bindings": {"C": "workspace:split_k2_partials"},
                "component": "cudnn_split_k2_partial_f32",
                "ordinal": 0,
            },
            {
                "bindings": {
                    "C": "output",
                    "elements": {
                        "lhs": {"name": "M", "op": "input"},
                        "op": "multiply",
                        "rhs": {"name": "N", "op": "input"},
                    },
                    "partials": "workspace:split_k2_partials",
                },
                "component": "cudnn_split_k2_reduce_bf16",
                "ordinal": 1,
            },
        ],
        "route": "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "same_stream": True,
    }
]
_WORKSPACES = [
    {
        "capture_requires_warmup": True,
        "dtype": "float32",
        "name": "split_k2_partials",
        "ownership": "device_stream_shape_private",
        "route": "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "shape": [
            2,
            {"name": "M", "op": "input"},
            {"name": "N", "op": "input"},
        ],
        "size_bytes": {
            "lhs": {"op": "constant", "value": 8},
            "op": "multiply",
            "rhs": {
                "lhs": {"name": "M", "op": "input"},
                "op": "multiply",
                "rhs": {"name": "N", "op": "input"},
            },
        },
    }
]
_VARIANT_KEYS = {
    "arg_plan",
    "cluster_dims",
    "component",
    "enable_pdl",
    "flat_grid",
    "grid_kind",
    "has_alpha",
    "kernel_symbol",
    "launch_grid",
    "module_ident",
    "output_dtype",
    "route",
    "schedule_symbol",
    "smem_bytes",
    "smem_data_offset_bytes",
    "smem_pool_bytes",
    "threads",
    "tma_descriptors",
    "use_pdl",
}
_COMPONENT_SPECS = {
    "cudnn_tma_bf16": (
        "prepared_native_tcgen05_cudnn_e4m3_tma_v1",
        "bfloat16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cudnn_tma_f16": (
        "prepared_native_tcgen05_cudnn_e4m3_tma_v1",
        "float16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cudnn_cp_async_bf16": (
        "prepared_native_tcgen05_cudnn_e4m3_cp_async_v1",
        "bfloat16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cudnn_cp_async_f16": (
        "prepared_native_tcgen05_cudnn_e4m3_cp_async_v1",
        "float16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cute_bf16": (
        "prepared_native_tcgen05_cute_dsl_s0e5m3_v1",
        "bfloat16",
        (("generic_2d", False), ("generic_flat", True)),
        None,
    ),
    "cudnn_group_m128_bf16": (
        "prepared_native_tcgen05_cudnn_e4m3_group_m128_v1",
        "bfloat16",
        (("group_m128_2d", False),),
        None,
    ),
    "cudnn_split_k2_partial_f32": (
        "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "float32",
        (("split_k2_partial", False),),
        None,
    ),
    "cudnn_split_k2_reduce_bf16": (
        "prepared_native_tcgen05_cudnn_e4m3_split_k2_v1",
        "bfloat16",
        (("split_k2_reduce", False),),
        None,
    ),
    "cute_warp_mma_m16_k16_bf16": (
        "prepared_native_warp_mma_m16n64k16_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        16,
    ),
    "cute_warp_mma_m16_k32_bf16": (
        "prepared_native_warp_mma_m16n64k32_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        16,
    ),
    "cute_warp_mma_m16_k48_bf16": (
        "prepared_native_warp_mma_m16n64k48_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        16,
    ),
    "cute_warp_mma_m16_bf16": (
        "prepared_native_warp_mma_m16n64k128_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        16,
    ),
    "cute_warp_mma_m32_bf16": (
        "prepared_native_warp_mma_m32n64k128_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        32,
    ),
    "cute_warp_mma_m64_bf16": (
        "prepared_native_warp_mma_m64n64k128_cute_dsl_s0e5m3_f16mma_v2",
        "bfloat16",
        (("persistent_sm_count", False),),
        64,
    ),
}
_COMPONENT_ENUMS = {
    "cudnn_tma_bf16": "Component::kNativeTmaBf16",
    "cudnn_tma_f16": "Component::kNativeTmaF16",
    "cudnn_cp_async_bf16": "Component::kNativeCpAsyncBf16",
    "cudnn_cp_async_f16": "Component::kNativeCpAsyncF16",
    "cute_bf16": "Component::kTiledBaseBf16",
    "cudnn_group_m128_bf16": "Component::kNativeGroupM128Bf16",
    "cudnn_split_k2_partial_f32": "Component::kNativeSplitK2PartialF32",
    "cudnn_split_k2_reduce_bf16": "Component::kNativeSplitK2ReduceBf16",
    "cute_warp_mma_m16_k16_bf16": "Component::kTiledWarpM16K16Bf16",
    "cute_warp_mma_m16_k32_bf16": "Component::kTiledWarpM16K32Bf16",
    "cute_warp_mma_m16_k48_bf16": "Component::kTiledWarpM16K48Bf16",
    "cute_warp_mma_m16_bf16": "Component::kTiledWarpM16Bf16",
    "cute_warp_mma_m32_bf16": "Component::kTiledWarpM32Bf16",
    "cute_warp_mma_m64_bf16": "Component::kTiledWarpM64Bf16",
}
_INTEGRATION_KERNEL_KEYS = {
    "arg_plan",
    "arg_plan_kind",
    "cluster_dims",
    "component",
    "enable_pdl",
    "flat_grid",
    "grid_mode",
    "logical_grid_mode",
    "has_alpha",
    "ir_symbol",
    "kernel_symbol",
    "launch_grid",
    "module_ident",
    "output_dtype",
    "prepared_abi",
    "route",
    "schedule_symbol",
    "smem_bytes",
    "smem_data_offset_bytes",
    "smem_pool_bytes",
    "stage",
    "threads",
    "tma_descriptors",
    "use_pdl",
}
_INTEGRATION_ROW7_EXACT_SHAPE = {"M": 17, "N": 3072, "K": 3072}
_INTEGRATION_GRID_KINDS = {
    "two_dimensional": "generic_2d",
    "flat_overflow": "generic_flat",
    "group_m128": "group_m128_2d",
    "split_k2_partial": "split_k2_partial",
    "split_k2_reduce": "split_k2_reduce",
    "persistent": "persistent_sm_count",
    "persistent_m16_2sm": "persistent_sm_count",
}
_INTEGRATION_COMPONENT_METADATA = {
    "cudnn_tma_bf16": ("cudnn_tma", "cudnn", "compute"),
    "cudnn_tma_f16": ("cudnn_tma", "cudnn", "compute"),
    "cudnn_cp_async_bf16": ("cudnn_cp_async", "cudnn", "compute"),
    "cudnn_cp_async_f16": ("cudnn_cp_async", "cudnn", "compute"),
    "cute_bf16": ("cute_tma", "cute_dsl", "compute"),
    "cudnn_group_m128_bf16": ("cudnn_tma", "cudnn", "compute"),
    "cudnn_split_k2_partial_f32": ("cudnn_tma", "cudnn", "partial"),
    "cudnn_split_k2_reduce_bf16": ("split_k2_reduce", "workspace", "reduce"),
    "cute_warp_mma_m16_k16_bf16": ("cute_warp_short", "cute_dsl", "compute"),
    "cute_warp_mma_m16_k32_bf16": ("cute_warp_short", "cute_dsl", "compute"),
    "cute_warp_mma_m16_k48_bf16": ("cute_warp_short", "cute_dsl", "compute"),
    "cute_warp_mma_m16_bf16": ("cute_warp", "cute_dsl", "compute"),
    "cute_warp_mma_m32_bf16": ("cute_warp", "cute_dsl", "compute"),
    "cute_warp_mma_m64_bf16": ("cute_warp", "cute_dsl", "compute"),
}
_INTEGRATION_LAUNCH_RESOURCES = {
    "cudnn_tma_bf16": (512, 107520),
    "cudnn_tma_f16": (512, 107520),
    "cudnn_cp_async_bf16": (512, 107520),
    "cudnn_cp_async_f16": (512, 107520),
    "cute_bf16": (512, 107520),
    "cudnn_group_m128_bf16": (512, 139264),
    "cudnn_split_k2_partial_f32": (512, 107520),
    "cudnn_split_k2_reduce_bf16": (128, 0),
    "cute_warp_mma_m16_k16_bf16": (96, 150528),
    "cute_warp_mma_m16_k32_bf16": (96, 150528),
    "cute_warp_mma_m16_k48_bf16": (96, 150528),
    "cute_warp_mma_m16_bf16": (96, 150528),
    "cute_warp_mma_m32_bf16": (160, 218112),
    "cute_warp_mma_m64_bf16": (160, 73728),
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MODULE_IDENT_SUFFIX_PATTERN = re.compile(r"^[0-9a-f]{10}$")
_CPP_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KERNEL_SYMBOL_PATTERN = re.compile(
    r"^kernel_flashinfer_bf16_fp4_[A-Za-z0-9_]+$"
)
_KERNEL_DEFINITION_PATTERN = re.compile(
    r"\b__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+"
    r"(kernel_flashinfer_bf16_fp4_[A-Za-z0-9_]+)\s*\("
)
_VARIANT_ABI_SHA256 = (
    "46fc96e77732fb4ad0c8a8b171e8339954ddb1342835bc0de01afe074a73b889"
)
_KERNEL_SPECS_MARKER = "FLASHINFER_BLACKWELL_BF16_FP4_KERNEL_SPECS"
_KERNEL_COUNT_MARKER = "FLASHINFER_BLACKWELL_BF16_FP4_KERNEL_COUNT"


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "blackwell_bf16_fp4"
    if installed.is_dir():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "blackwell_bf16_fp4"
    if checkout.is_dir():
        return checkout

    raise FileNotFoundError(
        "Blackwell BF16 x FP4 GEMM sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _target_for_capability(capability: tuple[int, int]) -> str:
    if capability == (10, 0):
        return "sm100"
    if capability == (10, 3):
        return "sm103"
    raise ValueError(
        "Blackwell BF16 x FP4 GEMM requires compute capability 10.0 or 10.3, "
        f"got {capability[0]}.{capability[1]}"
    )


def _target() -> str:
    return _target_for_capability(torch.cuda.get_device_capability())


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate ABI manifest key {key!r}")
        result[key] = value
    return result


def _expected_variant_signatures() -> set[tuple[Any, ...]]:
    signatures: set[tuple[Any, ...]] = set()
    for component, (
        route,
        output_dtype,
        grids,
        tile_m,
    ) in _COMPONENT_SPECS.items():
        if component == "cudnn_split_k2_reduce_bf16":
            alpha_values: tuple[bool | None, ...] = (None,)
            reused_alpha: tuple[bool, ...] | None = (False, True)
        else:
            alpha_values = (False, True)
            reused_alpha = None
        for grid_kind, flat_grid in grids:
            for has_alpha in alpha_values:
                for enable_pdl in (False, True):
                    signatures.add(
                        (
                            component,
                            route,
                            output_dtype,
                            has_alpha,
                            enable_pdl,
                            grid_kind,
                            flat_grid,
                            tile_m,
                            reused_alpha,
                        )
                    )
    return signatures


_EXPECTED_VARIANT_SIGNATURES = _expected_variant_signatures()


def _variant_symbol_stem(variant: dict[str, Any]) -> str:
    stem = f"flashinfer_bf16_fp4_{variant['component']}"
    if variant["flat_grid"]:
        stem += "_flat"
    if variant["component"] == "cudnn_split_k2_reduce_bf16":
        return f"{stem}_pdl{int(variant['enable_pdl'])}"
    return (
        f"{stem}_a{int(variant['has_alpha'])}"
        f"_pdl{int(variant['enable_pdl'])}"
    )


def _variant_abi_sha256(variants: list[dict[str, Any]]) -> str:
    records = [
        {key: value for key, value in variant.items() if key != "module_ident"}
        for variant in variants
    ]
    records.sort(
        key=lambda record: json.dumps(
            record, sort_keys=True, separators=(",", ":")
        )
    )
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_variants(variants: Any) -> None:
    if not isinstance(variants, list) or len(variants) != 74:
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires 74 variants")

    signatures: set[tuple[Any, ...]] = set()
    kernel_symbols: set[str] = set()
    module_idents: set[str] = set()
    schedule_symbols: set[str] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid variant record"
            )

        component = variant.get("component")
        if not isinstance(component, str) or component not in _COMPONENT_SPECS:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an unknown component"
            )
        expected_keys = set(_VARIANT_KEYS)
        tile_m = _COMPONENT_SPECS[component][3]
        if tile_m is not None:
            expected_keys.add("tile_m")
        if component == "cudnn_split_k2_reduce_bf16":
            expected_keys.add("reuses_alpha_specializations")
        if set(variant) != expected_keys:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest variant keys do not match schema 3"
            )

        for key in ("kernel_symbol", "module_ident", "schedule_symbol"):
            value = variant[key]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    "Blackwell BF16 x FP4 ABI manifest has an invalid variant symbol"
                )
        if variant["kernel_symbol"] in kernel_symbols:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate kernel symbols"
            )
        if variant["module_ident"] in module_idents:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate module identifiers"
            )
        if variant["schedule_symbol"] in schedule_symbols:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate schedule symbols"
            )
        kernel_symbols.add(variant["kernel_symbol"])
        module_idents.add(variant["module_ident"])
        schedule_symbols.add(variant["schedule_symbol"])

        if variant["cluster_dims"] != [1, 1, 1]:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest requires unit cluster dimensions"
            )
        if (
            not isinstance(variant["enable_pdl"], bool)
            or not isinstance(variant["flat_grid"], bool)
            or variant["use_pdl"] is not variant["enable_pdl"]
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has invalid launch flags"
            )
        if (
            type(variant["threads"]) is not int
            or variant["threads"] <= 0
            or any(
                type(variant[key]) is not int or variant[key] < 0
                for key in (
                    "smem_bytes",
                    "smem_data_offset_bytes",
                    "smem_pool_bytes",
                )
            )
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has invalid launch resources"
            )

        arg_plan = variant["arg_plan"]
        if not isinstance(arg_plan, list) or any(
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or entry[0] not in {"buffer", "grid", "parameter", "tma_buffer"}
            or not isinstance(entry[1], str)
            or not entry[1]
            for entry in arg_plan
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest variant is missing arg_plan"
            )

        descriptors = variant["tma_descriptors"]
        tma_arguments = [
            (index, index, resource)
            for index, (kind, resource) in enumerate(arg_plan)
            if kind == "tma_buffer"
        ]
        if (
            not isinstance(descriptors, list)
            or any(not isinstance(descriptor, dict) for descriptor in descriptors)
            or [
                (
                    descriptor.get("host_argument_index"),
                    descriptor.get("kernel_argument_index"),
                    descriptor.get("resource"),
                )
                for descriptor in descriptors
            ]
            != tma_arguments
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest TMA descriptors do not match "
                "pointer arguments"
            )

        launch_grid = variant["launch_grid"]
        grid_arguments = {
            resource.removeprefix("grid_"): index
            for index, (kind, resource) in enumerate(arg_plan)
            if kind == "grid"
        }
        if (
            not isinstance(launch_grid, dict)
            or set(launch_grid) != {"x", "y", "z"}
            or set(grid_arguments) != {"x", "y", "z"}
            or any(
                not isinstance(launch_grid[axis], dict)
                or launch_grid[axis].get("host_argument_index") != argument_index
                or not isinstance(launch_grid[axis].get("expression"), dict)
                for axis, argument_index in grid_arguments.items()
            )
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest launch grid does not match "
                "grid arguments"
            )

        reused_alpha = variant.get("reuses_alpha_specializations")
        if component == "cudnn_split_k2_reduce_bf16":
            if reused_alpha != [False, True]:
                raise ValueError(
                    "Blackwell BF16 x FP4 ABI manifest has invalid reused-alpha ABI"
                )
        elif reused_alpha is not None:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has unexpected reused-alpha ABI"
            )
        for key in ("route", "output_dtype", "grid_kind"):
            if not isinstance(variant[key], str):
                raise ValueError(
                    "Blackwell BF16 x FP4 ABI manifest has an invalid variant matrix"
                )
        if variant["has_alpha"] is not None and not isinstance(
            variant["has_alpha"], bool
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid alpha specialization"
            )
        symbol_stem = _variant_symbol_stem(variant)
        if variant["schedule_symbol"] != symbol_stem:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an incompatible schedule symbol"
            )
        if variant["kernel_symbol"] != f"kernel_{symbol_stem}":
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an incompatible kernel symbol"
            )
        module_prefix = f"{symbol_stem}_"
        module_suffix = variant["module_ident"].removeprefix(module_prefix)
        if (
            not variant["module_ident"].startswith(module_prefix)
            or _MODULE_IDENT_SUFFIX_PATTERN.fullmatch(module_suffix) is None
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an incompatible module "
                "identifier"
            )
        signature = (
            component,
            variant["route"],
            variant["output_dtype"],
            variant["has_alpha"],
            variant["enable_pdl"],
            variant["grid_kind"],
            variant["flat_grid"],
            variant.get("tile_m"),
            tuple(reused_alpha) if reused_alpha is not None else None,
        )
        if signature in signatures:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has duplicate variant "
                "specializations"
            )
        signatures.add(signature)

    if signatures != _EXPECTED_VARIANT_SIGNATURES:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest variant matrix does not match schema 3"
        )
    if _variant_abi_sha256(variants) != _VARIANT_ABI_SHA256:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest variant ABI does not match schema 3"
        )


def _integration_component(kernel: dict[str, Any]) -> str:
    component = kernel.get("component")
    if component is not None and component not in _COMPONENT_SPECS:
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest has an unknown component"
        )

    grid_kind = _INTEGRATION_GRID_KINDS.get(
        kernel.get("logical_grid_mode", kernel.get("grid_mode"))
    )
    output_dtype = kernel.get("output_dtype")
    if output_dtype == "float32_workspace":
        output_dtype = "float32"
    physical_tile_m = kernel.get("tile_m")
    logical_tile_m = (
        32
        if (
            kernel.get("route")
            == _COMPONENT_SPECS["cute_warp_mma_m32_bf16"][0]
            and kernel.get("logical_grid_mode") == "persistent"
        )
        else physical_tile_m
    )
    candidates = [
        name
        for name, (route, expected_dtype, grids, tile_m) in _COMPONENT_SPECS.items()
        if kernel.get("route") == route
        and output_dtype == expected_dtype
        and (grid_kind, kernel.get("flat_grid")) in grids
        and logical_tile_m == tile_m
    ]
    if len(candidates) != 1 or (
        component is not None and component != candidates[0]
    ):
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest kernel does not resolve "
            "to one logical component"
        )
    return candidates[0]


def _integration_arg_plan(arg_plan_kind: str) -> list[list[str]]:
    if arg_plan_kind == "split_k2_reduce":
        return [
            ["buffer", "partials"],
            ["buffer", "C"],
            ["parameter", "elements"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
    if arg_plan_kind == "raw_pointer":
        return [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "B_descale"],
            ["buffer", "alpha"],
            ["buffer", "C"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
    prepared_kind = "buffer" if arg_plan_kind == "cudnn_cp_async" else "tma_buffer"
    descale_kind = (
        "buffer"
        if arg_plan_kind in {"cudnn_cp_async", "cute_warp_short"}
        else "tma_buffer"
    )
    output_kind = (
        "tma_buffer"
        if arg_plan_kind in {"cute_warp", "cute_warp_short"}
        else "buffer"
    )
    return [
        ["tma_buffer", "A"],
        [prepared_kind, "B"],
        [descale_kind, "B_descale"],
        ["buffer", "alpha"],
        [output_kind, "C"],
        ["parameter", "M"],
        ["parameter", "N"],
        ["parameter", "K"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ]


def _expected_integration_arg_plan(component: str) -> list[list[str]]:
    return _integration_arg_plan(_INTEGRATION_COMPONENT_METADATA[component][0])


def _raw_pointer_integration_arg_plan() -> list[list[str]]:
    return _integration_arg_plan("raw_pointer")


def _is_m32_raw_pointer_kernel(
    component: str, kernel: dict[str, Any]
) -> bool:
    return (
        component == "cute_warp_mma_m32_bf16"
        and kernel.get("arg_plan_kind") == "raw_pointer"
        and kernel.get("grid_mode") == "persistent_m16_2sm"
    )


def _validate_integration_manifest(manifest: dict[str, Any]) -> None:
    if manifest["prepared_abis"] != _PREPARED_ABIS:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest has incompatible prepared layouts"
        )

    ir_symbols = manifest["ir_symbols"]
    if (
        not isinstance(ir_symbols, list)
        or len(ir_symbols) != 15
        or any(not isinstance(symbol, str) or not symbol for symbol in ir_symbols)
        or len(set(ir_symbols)) != len(ir_symbols)
    ):
        raise ValueError("Blackwell BF16 x FP4 ABI manifest has invalid IR symbols")

    kernels = manifest["kernels"]
    if not isinstance(kernels, list) or len(kernels) != 75:
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires 75 kernels")

    kernel_symbols: set[str] = set()
    module_idents: set[str] = set()
    schedule_symbols: set[str] = set()
    observed_ir_symbols: set[str] = set()
    observed_components: set[str] = set()
    exact_row7: list[dict[str, Any]] = []
    generic_row7: list[dict[str, Any]] = []

    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid kernel record"
            )
        expected_keys = set(_INTEGRATION_KERNEL_KEYS)
        if "component" not in kernel:
            expected_keys.remove("component")
        if "tile_m" in kernel:
            expected_keys.add("tile_m")
        if "exact_shape" in kernel:
            expected_keys.add("exact_shape")
        if set(kernel) != expected_keys:
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest kernel keys do not "
                "match schema 3"
            )

        for key, seen, pattern in (
            ("kernel_symbol", kernel_symbols, _KERNEL_SYMBOL_PATTERN),
            ("module_ident", module_idents, _CPP_IDENTIFIER_PATTERN),
            ("schedule_symbol", schedule_symbols, _CPP_IDENTIFIER_PATTERN),
        ):
            value = kernel[key]
            if (
                not isinstance(value, str)
                or pattern.fullmatch(value) is None
                or value in seen
            ):
                raise ValueError(
                    "Blackwell BF16 x FP4 ABI manifest has an invalid or duplicate "
                    f"{key}"
                )
            seen.add(value)

        ir_symbol = kernel["ir_symbol"]
        if not isinstance(ir_symbol, str) or not ir_symbol:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid IR symbol"
            )
        observed_ir_symbols.add(ir_symbol)

        if kernel["cluster_dims"] != [1, 1, 1]:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest requires unit cluster dimensions"
            )
        if (
            not isinstance(kernel["enable_pdl"], bool)
            or not isinstance(kernel["flat_grid"], bool)
            or kernel["use_pdl"] is not kernel["enable_pdl"]
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has invalid launch flags"
            )
        if kernel["has_alpha"] is not None and not isinstance(
            kernel["has_alpha"], bool
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid alpha specialization"
            )
        if (
            type(kernel["threads"]) is not int
            or kernel["threads"] <= 0
            or any(
                type(kernel[key]) is not int or kernel[key] < 0
                for key in (
                    "smem_bytes",
                    "smem_data_offset_bytes",
                    "smem_pool_bytes",
                )
            )
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has invalid launch resources"
            )

        arg_plan = kernel["arg_plan"]
        if not isinstance(arg_plan, list) or any(
            not isinstance(entry, list)
            or len(entry) != 2
            or any(not isinstance(value, str) or not value for value in entry)
            for entry in arg_plan
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest kernel is missing arg_plan"
            )
        descriptors = kernel["tma_descriptors"]
        if not isinstance(descriptors, list) or any(
            not isinstance(descriptor, dict) for descriptor in descriptors
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest kernel is missing TMA descriptors"
            )
        tma_arguments = [
            (index, resource)
            for index, (kind, resource) in enumerate(arg_plan)
            if kind == "tma_buffer"
        ]
        if [
            (descriptor.get("host_argument_index"), descriptor.get("resource"))
            for descriptor in descriptors
        ] != tma_arguments:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest TMA descriptors do not match "
                "pointer arguments"
            )

        component = _integration_component(kernel)
        observed_components.add(component)
        expected_arg_plan_kind, expected_prepared_abi, expected_stage = (
            _INTEGRATION_COMPONENT_METADATA[component]
        )
        expected_threads, expected_smem = _INTEGRATION_LAUNCH_RESOURCES[component]
        raw_pointer_m32 = _is_m32_raw_pointer_kernel(component, kernel)
        expected_arg_plan = (
            _raw_pointer_integration_arg_plan()
            if raw_pointer_m32
            else _expected_integration_arg_plan(component)
        )
        if arg_plan != expected_arg_plan:
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest kernel arg_plan "
                "does not match its logical component"
            )
        if not raw_pointer_m32 and (
            kernel["arg_plan_kind"] != expected_arg_plan_kind
            or kernel["prepared_abi"] != expected_prepared_abi
            or kernel["stage"] != expected_stage
            or (kernel["threads"], kernel["smem_bytes"])
            != (expected_threads, expected_smem)
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest kernel metadata "
                "does not match its logical component"
            )

        exact_shape = kernel.get("exact_shape")
        if exact_shape is not None:
            if (
                not isinstance(exact_shape, dict)
                or set(exact_shape) != {"M", "N", "K"}
                or any(
                    type(exact_shape[name]) is not int or exact_shape[name] <= 0
                    for name in ("M", "N", "K")
                )
                or exact_shape != _INTEGRATION_ROW7_EXACT_SHAPE
            ):
                raise ValueError(
                    "Blackwell BF16 x FP4 integration manifest row7 exact shape "
                    "does not match schema 3"
                )
        if exact_shape is not None and not raw_pointer_m32:
            raise ValueError(
                "Blackwell BF16 x FP4 integration manifest exact shape is not "
                "the row7 raw-pointer specialization"
            )
        if raw_pointer_m32:
            if (
                exact_shape is None
                or kernel["has_alpha"] is not True
                or kernel["enable_pdl"] is not True
                or kernel["logical_grid_mode"] != "persistent"
                or kernel.get("tile_m") != 16
                or (kernel["threads"], kernel["smem_bytes"]) != (128, 34816)
                or kernel["prepared_abi"] != "cute_dsl"
                or kernel["stage"] != "compute"
                or kernel["smem_data_offset_bytes"] != 0
                or kernel["smem_pool_bytes"] != 34816
                or descriptors
            ):
                raise ValueError(
                    "Blackwell BF16 x FP4 integration manifest row7 exact shape "
                    "does not match schema 3"
                )
            exact_row7.append(kernel)
        elif (
            component == "cute_warp_mma_m32_bf16"
            and kernel["has_alpha"] is True
            and kernel["enable_pdl"] is True
        ):
            generic_row7.append(kernel)

    if observed_ir_symbols != set(ir_symbols):
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest IR inventory is incomplete"
        )
    if observed_components != set(_COMPONENT_SPECS):
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest component inventory is incomplete"
        )
    if len(exact_row7) != 1 or len(generic_row7) != 1:
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest requires one exact row7 "
            "specialization and one generic M32 fallback"
        )
    if sum(
        kernel["ir_symbol"] == exact_row7[0]["ir_symbol"] for kernel in kernels
    ) != 1:
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest physical IR inventory "
            "does not isolate the exact row7 specialization"
        )

    dispatch = manifest["dispatch"]
    if not isinstance(dispatch, dict):
        raise ValueError("Blackwell BF16 x FP4 ABI manifest dispatch must be an object")
    if dispatch.get("selection") != "ordered_first_match_after_input_validation":
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest has incompatible dispatch ordering"
        )
    if dispatch.get("inputs") != _DISPATCH_INPUTS:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest has incompatible dispatch inputs"
        )
    routes = dispatch.get("routes")
    if not isinstance(routes, list) or len(routes) != 11:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest requires 11 dispatch routes"
        )
    m32_route_name = _COMPONENT_SPECS["cute_warp_mma_m32_bf16"][0]
    m32_routes = [route for route in routes if route.get("route") == m32_route_name]
    if len(m32_routes) != 1:
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest is missing the M32 route"
        )
    specializations = m32_routes[0].get("specializations")
    exact_match = {
        "out_dtype": "bfloat16",
        "has_alpha": True,
        "enable_pdl": True,
        **_INTEGRATION_ROW7_EXACT_SHAPE,
    }
    generic_match = {
        "out_dtype": "bfloat16",
        "has_alpha": True,
        "enable_pdl": True,
    }
    if (
        not isinstance(specializations, list)
        or len(specializations) != 5
        or specializations[0].get("match") != exact_match
        or not any(
            specialization.get("match") == generic_match
            for specialization in specializations[1:]
        )
    ):
        raise ValueError(
            "Blackwell BF16 x FP4 integration manifest row7 exact route must "
            "precede the generic M32 fallback"
        )


def _manifest_kernel_specs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("kernels", manifest.get("variants"))
    if not isinstance(records, list):
        raise ValueError("Blackwell BF16 x FP4 ABI manifest has no kernel inventory")
    integration = "kernels" in manifest
    specs = []
    for record in records:
        component = _integration_component(record) if integration else record["component"]
        specs.append(
            {
                "component": component,
                "has_alpha": bool(record.get("has_alpha", False)),
                "enable_pdl": record["enable_pdl"],
                "flat_grid": record["flat_grid"],
                "kernel_symbol": record["kernel_symbol"],
                "threads": record["threads"],
                "smem_bytes": record["smem_bytes"],
                "raw_pointer_abi": integration
                and record.get("arg_plan_kind") == "raw_pointer",
                "persistent_m16_2sm": integration
                and record.get("grid_mode") == "persistent_m16_2sm",
                "exact_m": record.get("exact_shape", {}).get("M", 0),
                "exact_n": record.get("exact_shape", {}).get("N", 0),
                "exact_k": record.get("exact_shape", {}).get("K", 0),
            }
        )
    return specs


def _render_binding_source(
    binding_raw: bytes,
    manifest: dict[str, Any],
    module_ident: str,
) -> str:
    try:
        source = binding_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Blackwell BF16 x FP4 binding source must be UTF-8") from error
    if source.count(_KERNEL_SPECS_MARKER) != 1:
        raise ValueError(
            "Blackwell BF16 x FP4 binding must contain exactly one kernel-spec marker"
        )
    if source.count(_KERNEL_COUNT_MARKER) != 1:
        raise ValueError(
            "Blackwell BF16 x FP4 binding must contain exactly one kernel-count marker"
        )
    if (
        _CPP_IDENTIFIER_PATTERN.fullmatch(module_ident) is None
        or "FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT" not in source
    ):
        raise ValueError("Blackwell BF16 x FP4 binding module identifier is invalid")

    specs = _manifest_kernel_specs(manifest)
    rendered_specs = ",\n    ".join(
        "KernelSpec{{{component}, {has_alpha}, {enable_pdl}, {flat_grid}, "
        "{raw_pointer_abi}, {persistent_m16_2sm}, "
        "{exact_m}, {exact_n}, {exact_k}, "
        '\"{kernel_symbol}\", {threads}u, {smem_bytes}u}}'.format(
            component=_COMPONENT_ENUMS[record["component"]],
            has_alpha=str(record["has_alpha"]).lower(),
            enable_pdl=str(record["enable_pdl"]).lower(),
            flat_grid=str(record["flat_grid"]).lower(),
            raw_pointer_abi=str(record["raw_pointer_abi"]).lower(),
            persistent_m16_2sm=str(record["persistent_m16_2sm"]).lower(),
            exact_m=record["exact_m"],
            exact_n=record["exact_n"],
            exact_k=record["exact_k"],
            kernel_symbol=record["kernel_symbol"],
            threads=record["threads"],
            smem_bytes=record["smem_bytes"],
        )
        for record in specs
    )
    return (
        source.replace(_KERNEL_SPECS_MARKER, rendered_specs)
        .replace(_KERNEL_COUNT_MARKER, str(len(specs)))
        .replace("FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT", module_ident)
    )


def _manifest_kernel_symbols(manifest: dict[str, Any]) -> set[str]:
    if "kernels" in manifest:
        return {kernel["kernel_symbol"] for kernel in manifest["kernels"]}
    return {variant["kernel_symbol"] for variant in manifest["variants"]}


def _load_abi_manifest(path: Path, target: str) -> tuple[dict[str, Any], bytes]:
    if target not in _MANIFEST_NAMES:
        raise ValueError(f"unknown Blackwell BF16 x FP4 target {target!r}")

    raw = path.read_bytes()
    try:
        manifest = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid Blackwell BF16 x FP4 ABI manifest {path.name}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("Blackwell BF16 x FP4 ABI manifest root must be an object")

    keys = set(manifest)
    if keys == _INTEGRATION_MANIFEST_KEYS:
        manifest_family = "integration"
    elif keys == _VARIANT_MANIFEST_KEYS:
        manifest_family = "variant"
    else:
        expected_keys = (
            _INTEGRATION_MANIFEST_KEYS
            if keys & {"prepared_abis", "ir_symbols", "kernels", "dispatch"}
            else _VARIANT_MANIFEST_KEYS
        )
        missing = sorted(expected_keys - keys)
        unexpected = sorted(keys - expected_keys)
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest keys do not match schema 3; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if manifest["schema_version"] != 3:
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires schema_version=3")
    if manifest["bundle"] != "flashinfer_blackwell_bf16_fp4_gemm":
        raise ValueError("Blackwell BF16 x FP4 ABI manifest has an unexpected bundle")
    if manifest["arch"] != _NVCC_ARCH[target]:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest architecture does not match "
            f"{target}: {manifest['arch']!r}"
        )
    if manifest["tma_abi"] != "pointer":
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires pointer TMA ABI")
    if manifest["tensor_map_abi"] != _TENSOR_MAP_ABI:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest has an incompatible TensorMap ABI"
        )
    if manifest["adapter_boundary"] != "separate_translation_unit":
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest requires a separate adapter "
            "translation unit"
        )
    if manifest_family == "integration":
        _validate_integration_manifest(manifest)
    else:
        if manifest["dispatcher"] != _DISPATCHER:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has incompatible dispatch routing"
            )
        if manifest["composite_routes"] != _COMPOSITE_ROUTES:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has incompatible composite routing"
            )
        if manifest["workspaces"] != _WORKSPACES:
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has incompatible workspace ABI"
            )
        _validate_variants(manifest["variants"])

    return manifest, raw


def _source_define(source: str, name: str) -> str:
    match = re.search(rf"^#define {re.escape(name)}\s+(.+?)\s*$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"generated Blackwell BF16 x FP4 source is missing {name}")
    return match.group(1)


def _validate_source_header(
    source_raw: bytes,
    manifest: dict[str, Any],
    manifest_raw: bytes,
    target: str,
) -> None:
    try:
        source = source_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            "generated Blackwell BF16 x FP4 source must be UTF-8"
        ) from error

    if _source_define(source, "FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY") != "1":
        raise ValueError("generated Blackwell BF16 x FP4 source is not marked ready")
    if _source_define(source, "FLASHINFER_BLACKWELL_BF16_FP4_ABI_VERSION") != "3":
        raise ValueError(
            "generated Blackwell BF16 x FP4 source has an incompatible ABI version"
        )
    if _source_define(source, "FLASHINFER_BLACKWELL_BF16_FP4_TARGET_SM") != str(
        _TARGET_SM[target]
    ):
        raise ValueError(
            "generated Blackwell BF16 x FP4 source target does not match manifest"
        )

    raw_source_sha256 = _source_define(
        source, "FLASHINFER_BLACKWELL_BF16_FP4_RAW_SOURCE_SHA256"
    ).strip('"')
    if _SHA256_PATTERN.fullmatch(raw_source_sha256) is None:
        raise ValueError(
            "generated Blackwell BF16 x FP4 source has an invalid source hash"
        )
    manifest_sha256 = _source_define(
        source, "FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256"
    ).strip('"')
    if manifest_sha256 != hashlib.sha256(manifest_raw).hexdigest():
        raise ValueError(
            "generated Blackwell BF16 x FP4 source does not match its ABI manifest"
        )

    source_symbols = set(_KERNEL_DEFINITION_PATTERN.findall(source))
    manifest_symbols = _manifest_kernel_symbols(manifest)
    if source_symbols != manifest_symbols:
        raise ValueError(
            "generated Blackwell BF16 x FP4 source kernel symbols do not match "
            "its ABI manifest"
        )


def _source_package_key(
    target: str,
    source_raw: bytes,
    manifest_raw: bytes,
    binding_raw: bytes,
    nvcc: Path,
) -> str:
    digest = hashlib.sha256()
    for part in (
        source_raw,
        manifest_raw,
        binding_raw,
        target.encode(),
        str(nvcc).encode(),
    ):
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()[:16]


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError("nvcc is required to build Blackwell BF16 x FP4 GEMM")
    return Path(candidate).resolve()


def _copy_if_different(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.read_bytes() == source.read_bytes():
        return
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


@functools.cache
def _load_module(target: str):
    if target not in _SOURCE_NAMES:
        raise ValueError(f"unknown Blackwell BF16 x FP4 target {target!r}")
    source_dir = _source_dir()
    generated_source = source_dir / _SOURCE_NAMES[target]
    manifest_path = source_dir / _MANIFEST_NAMES[target]
    binding_source = source_dir / _BINDING_NAME
    source_package = (generated_source, manifest_path, binding_source)
    missing = [path.name for path in source_package if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Blackwell BF16 x FP4 GEMM source package is incomplete; missing: "
            + ", ".join(missing)
        )

    source_raw = generated_source.read_bytes()
    manifest, manifest_raw = _load_abi_manifest(manifest_path, target)
    binding_raw = binding_source.read_bytes()
    _validate_source_header(source_raw, manifest, manifest_raw, target)

    nvcc = _nvcc()
    key = _source_package_key(target, source_raw, manifest_raw, binding_raw, nvcc)
    module_ident = f"flashinfer_blackwell_bf16_fp4_{target}_{key}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_ident
    build_dir.mkdir(parents=True, exist_ok=True)

    local_generated_source = build_dir / generated_source.name
    local_manifest = build_dir / manifest_path.name
    local_binding_source = build_dir / binding_source.name
    _copy_if_different(generated_source, local_generated_source)
    _copy_if_different(manifest_path, local_manifest)
    _copy_if_different(binding_source, local_binding_source)

    cubin_path = build_dir / f"{module_ident}.cubin"
    if not cubin_path.is_file():
        temporary_cubin = build_dir / f"{module_ident}.{os.getpid()}.tmp.cubin"
        command = [
            str(nvcc),
            "-cubin",
            f"-arch={_NVCC_ARCH[target]}",
            "--std=c++17",
            "-O3",
            "--use_fast_math",
            "-I",
            str(nvcc.parent.parent / "include"),
            str(local_generated_source),
            "-o",
            str(temporary_cubin),
        ]
        process = subprocess.run(command, text=True, capture_output=True)
        if process.returncode != 0:
            temporary_cubin.unlink(missing_ok=True)
            raise RuntimeError(
                "Blackwell BF16 x FP4 GEMM nvcc failed for "
                f"{_NVCC_ARCH[target]}:\n{process.stderr}"
            )
        os.replace(temporary_cubin, cubin_path)

    host_source = _render_binding_source(
        binding_raw,
        manifest,
        module_ident,
    )
    return cpp.load_inline(
        module_ident,
        cpp_sources=host_source,
        embed_cubin={module_ident: cubin_path.read_bytes()},
        extra_include_paths=[str(nvcc.parent.parent / "include")],
        extra_cflags=[
            "-O3",
            f"-DFLASHINFER_BLACKWELL_BF16_FP4_TARGET_MINOR={_TARGET_MINOR[target]}",
        ],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_dir),
    )


def get_blackwell_bf16_fp4_module():
    """Return the JIT module compiled for the current SM100-family target."""

    return _load_module(_target())


__all__ = ["get_blackwell_bf16_fp4_module"]
