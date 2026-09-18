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

MsaVibeCudaTarget = Literal["sm100a", "sm103a"]

_MSA_VIBECUDA_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

_MSA_VIBECUDA_SOURCES = (
    "msa_vibecuda_binding.cu",
    "msa_vibecuda_core.cu",
    "msa_vibecuda_g16.cu",
    "msa_vibecuda_g4.cu",
)


def _get_msa_vibecuda_csrc_dir() -> Path:
    """Locate the VibeCUDA MSA CUDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "msa_vibecuda"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "msa_vibecuda"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "VibeCUDA MSA CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_msa_vibecuda_include_dir() -> Path:
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


def get_msa_vibecuda_uri(target: MsaVibeCudaTarget) -> str:
    """Return the target-specific JIT/AOT key for the VibeCUDA MSA module."""

    if target not in _MSA_VIBECUDA_NVCC_FLAGS:
        raise ValueError(f"unsupported VibeCUDA MSA target: {target}")
    return f"msa_vibecuda_{target}"


@functools.cache
def gen_msa_vibecuda_module(target: MsaVibeCudaTarget) -> JitSpec:
    """Generate the SM100a or SM103a VibeCUDA MSA JIT module."""

    csrc_dir = _get_msa_vibecuda_csrc_dir()
    uri = get_msa_vibecuda_uri(target)

    sources = []
    for fname in _MSA_VIBECUDA_SOURCES:
        body = csrc_dir / fname
        if not body.is_file():
            raise FileNotFoundError(f"VibeCUDA MSA CUDA source not found: {body}")
        sources.append(body)

    spec = gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=[
            "-O3",
            # Matches the validated level-3 build: the HMMA fallback softmax
            # path relies on fast exp2/div lowering for its measured perf.
            "--use_fast_math",
            *_MSA_VIBECUDA_NVCC_FLAGS[target],
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_msa_vibecuda_include_dir(),
        ],
    )
    logger.info(f"Generated VibeCUDA MSA {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def get_msa_vibecuda_module(target: MsaVibeCudaTarget):
    """Build or load the VibeCUDA MSA module for one SM100 target."""

    module = gen_msa_vibecuda_module(target).build_and_load()
    logger.info(f"Loaded VibeCUDA MSA {target} module")
    return module


__all__ = [
    "MsaVibeCudaTarget",
    "gen_msa_vibecuda_module",
    "get_msa_vibecuda_module",
    "get_msa_vibecuda_uri",
]
