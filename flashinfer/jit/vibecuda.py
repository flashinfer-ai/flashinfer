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

import functools
from pathlib import Path
from typing import Literal

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

VibeCUDABSATarget = Literal["sm100a", "sm103a"]

_VIBECUDA_BSA_NVCC_FLAGS: dict[VibeCUDABSATarget, list[str]] = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}


def _get_vibecuda_bsa_csrc_dir() -> Path:
    """Locate the VibeCUDA BSA CUDA sources in installed and source checkouts."""

    if jit_env.FLASHINFER_CSRC_DIR.exists():
        return jit_env.FLASHINFER_CSRC_DIR

    checkout = Path(__file__).resolve().parents[2] / "csrc"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "VibeCUDA BSA CUDA sources were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_CSRC_DIR}\n"
        f"  - {checkout}"
    )


def _get_vibecuda_bsa_include_dir() -> Path:
    """Locate FlashInfer headers in installed and source checkouts."""

    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR

    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def get_vibecuda_bsa_uri(target: VibeCUDABSATarget) -> str:
    """Return the target-specific JIT/AOT key for the VibeCUDA BSA module."""

    if target not in _VIBECUDA_BSA_NVCC_FLAGS:
        raise ValueError(f"unsupported VibeCUDA BSA target: {target}")
    return f"vibecuda_bsa_{target}"


@functools.cache
def gen_vibecuda_bsa_module(target: VibeCUDABSATarget) -> JitSpec:
    """Generate the exact-SM100a or exact-SM103a VibeCUDA BSA JIT module."""

    csrc_dir = _get_vibecuda_bsa_csrc_dir()
    include_dir = _get_vibecuda_bsa_include_dir()
    uri = get_vibecuda_bsa_uri(target)
    body = csrc_dir / "vibecuda_bsa.cu"
    binding = csrc_dir / "vibecuda_bsa_jit_binding.cu"
    if not body.exists():
        raise FileNotFoundError(f"VibeCUDA BSA CUDA source not found: {body}")
    if not binding.exists():
        raise FileNotFoundError(f"VibeCUDA BSA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=uri,
        sources=[body, binding],
        extra_cuda_cflags=[*_VIBECUDA_BSA_NVCC_FLAGS[target]],
        extra_include_paths=[csrc_dir, include_dir],
    )
    logger.info(f"Generated VibeCUDA BSA {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def load_vibecuda_bsa_module(target: VibeCUDABSATarget):
    """Build or load the physical, target-specific VibeCUDA BSA module."""

    module = gen_vibecuda_bsa_module(target).build_and_load()
    logger.info(f"Loaded VibeCUDA BSA {target} module")
    return module


def get_vibecuda_bsa_module(target: VibeCUDABSATarget):
    """Return the loaded module used by the VibeCUDA BSA backend."""

    return load_vibecuda_bsa_module(target)


__all__ = [
    "VibeCUDABSATarget",
    "gen_vibecuda_bsa_module",
    "get_vibecuda_bsa_module",
    "get_vibecuda_bsa_uri",
    "load_vibecuda_bsa_module",
]
