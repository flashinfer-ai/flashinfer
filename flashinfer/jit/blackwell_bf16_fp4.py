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

_MANIFEST_KEYS = {
    "schema_version",
    "bundle",
    "arch",
    "tma_abi",
    "tensor_map_abi",
    "adapter_boundary",
    "prepared_abis",
    "ir_symbols",
    "kernels",
    "dispatch",
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
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
    if keys != _MANIFEST_KEYS:
        missing = sorted(_MANIFEST_KEYS - keys)
        unexpected = sorted(keys - _MANIFEST_KEYS)
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
            "Blackwell BF16 x FP4 ABI manifest requires a separate adapter translation unit"
        )
    if manifest["prepared_abis"] != _PREPARED_ABIS:
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest has incompatible prepared layouts"
        )

    ir_symbols = manifest["ir_symbols"]
    if (
        not isinstance(ir_symbols, list)
        or len(ir_symbols) != 14
        or any(not isinstance(symbol, str) or not symbol for symbol in ir_symbols)
        or len(set(ir_symbols)) != len(ir_symbols)
    ):
        raise ValueError("Blackwell BF16 x FP4 ABI manifest has invalid IR symbols")

    kernels = manifest["kernels"]
    if not isinstance(kernels, list) or len(kernels) != 74:
        raise ValueError("Blackwell BF16 x FP4 ABI manifest requires 74 kernels")
    kernel_symbols = []
    for kernel in kernels:
        if not isinstance(kernel, dict) or not isinstance(
            kernel.get("kernel_symbol"), str
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest has an invalid kernel record"
            )
        arg_plan = kernel.get("arg_plan")
        if not isinstance(arg_plan, list) or any(
            not isinstance(entry, list)
            or len(entry) != 2
            or any(not isinstance(value, str) or not value for value in entry)
            for entry in arg_plan
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest kernel is missing arg_plan"
            )
        descriptors = kernel.get("tma_descriptors")
        if not isinstance(descriptors, list):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest kernel is missing TMA descriptors"
            )
        tma_arguments = [
            (index, resource)
            for index, (kind, resource) in enumerate(arg_plan)
            if kind == "tma_buffer"
        ]
        if (
            any(not isinstance(descriptor, dict) for descriptor in descriptors)
            or [
                (descriptor.get("host_argument_index"), descriptor.get("resource"))
                for descriptor in descriptors
            ]
            != tma_arguments
        ):
            raise ValueError(
                "Blackwell BF16 x FP4 ABI manifest TMA descriptors do not match pointer arguments"
            )
        kernel_symbols.append(kernel["kernel_symbol"])
    if len(set(kernel_symbols)) != len(kernel_symbols):
        raise ValueError(
            "Blackwell BF16 x FP4 ABI manifest has duplicate kernel symbols"
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

    return manifest, raw


def _source_define(source: str, name: str) -> str:
    match = re.search(rf"^#define {re.escape(name)}\s+(.+?)\s*$", source, re.MULTILINE)
    if match is None:
        raise ValueError(f"generated Blackwell BF16 x FP4 source is missing {name}")
    return match.group(1)


def _validate_source_header(
    source_raw: bytes, manifest_raw: bytes, target: str
) -> None:
    try:
        source = source_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            "generated Blackwell BF16 x FP4 source must be UTF-8"
        ) from error

    if _source_define(source, "FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY") != "1":
        raise ValueError("generated Blackwell BF16 x FP4 source is not marked ready")
    if _source_define(source, "FLASHINFER_BLACKWELL_BF16_FP4_ABI_VERSION") != "2":
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
    _, manifest_raw = _load_abi_manifest(manifest_path, target)
    binding_raw = binding_source.read_bytes()
    _validate_source_header(source_raw, manifest_raw, target)

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

    host_source = local_binding_source.read_text(encoding="utf-8").replace(
        "FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT", module_ident
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
