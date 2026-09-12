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
import os
import shutil
import subprocess
from pathlib import Path

import torch
from tvm_ffi import cpp

from . import env as jit_env


_SOURCE_NAMES = {
    "sm100": "flashinfer_blackwell_bf16_fp4_generated_sm100.cu",
    "sm103": "flashinfer_blackwell_bf16_fp4_generated_sm103.cu",
}
_NVCC_ARCH = {"sm100": "sm_100a", "sm103": "sm_103a"}
_TARGET_MINOR = {"sm100": 0, "sm103": 3}
_BINDING_NAME = "flashinfer_blackwell_bf16_fp4_binding.cu"
_KERNEL_SPECS_NAME = "flashinfer_blackwell_bf16_fp4_kernel_specs.h"


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


def _source_package_key(
    target: str,
    source_raw: bytes,
    kernel_specs_raw: bytes,
    binding_raw: bytes,
    nvcc: Path,
) -> str:
    digest = hashlib.sha256()
    for part in (
        source_raw,
        kernel_specs_raw,
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
    kernel_specs = source_dir / _KERNEL_SPECS_NAME
    binding_source = source_dir / _BINDING_NAME
    source_package = (generated_source, kernel_specs, binding_source)
    missing = [path.name for path in source_package if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Blackwell BF16 x FP4 GEMM source package is incomplete; missing: "
            + ", ".join(missing)
        )

    source_raw = generated_source.read_bytes()
    kernel_specs_raw = kernel_specs.read_bytes()
    binding_raw = binding_source.read_bytes()

    nvcc = _nvcc()
    key = _source_package_key(target, source_raw, kernel_specs_raw, binding_raw, nvcc)
    module_ident = f"flashinfer_blackwell_bf16_fp4_{target}_{key}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_ident
    build_dir.mkdir(parents=True, exist_ok=True)

    local_generated_source = build_dir / generated_source.name
    local_kernel_specs = build_dir / kernel_specs.name
    local_binding_source = build_dir / binding_source.name
    _copy_if_different(generated_source, local_generated_source)
    _copy_if_different(kernel_specs, local_kernel_specs)
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

    host_source = binding_raw.decode("utf-8").replace(
        "FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT", module_ident
    )
    return cpp.load_inline(
        module_ident,
        cpp_sources=host_source,
        embed_cubin={module_ident: cubin_path.read_bytes()},
        extra_include_paths=[str(build_dir), str(nvcc.parent.parent / "include")],
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
