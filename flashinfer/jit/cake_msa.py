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
    sm100f_nvcc_flags,
)

CakeMSAVariant = Literal[
    "decode_bf16_flat",
    "decode_bf16_paged",
    "decode_fp16_flat",
    "decode_fp16_paged",
    "decode_fp8_flat",
    "decode_fp8_paged",
    "decode_m16_bf16_flat",
    "decode_m16_bf16_paged",
    "prefill_m128_bf16_flat",
    "prefill_m128_bf16_gqa16_flat",
    "prefill_m128_bf16_gqa16_paged",
    "prefill_m128_bf16_paged",
    "prefill_m128_fp16_flat",
    "prefill_m128_fp16_paged",
    "prefill_m128_fp8_flat",
    "prefill_m128_fp8_paged",
    "prefill_m64_bf16_flat",
    "topk",
]
CakeMSATarget = Literal["sm100a", "sm100f"]

CAKE_MSA_VARIANTS: tuple[CakeMSAVariant, ...] = (
    "decode_bf16_flat",
    "decode_bf16_paged",
    "decode_fp16_flat",
    "decode_fp16_paged",
    "decode_fp8_flat",
    "decode_fp8_paged",
    "decode_m16_bf16_flat",
    "decode_m16_bf16_paged",
    "prefill_m128_bf16_flat",
    "prefill_m128_bf16_gqa16_flat",
    "prefill_m128_bf16_gqa16_paged",
    "prefill_m128_bf16_paged",
    "prefill_m128_fp16_flat",
    "prefill_m128_fp16_paged",
    "prefill_m128_fp8_flat",
    "prefill_m128_fp8_paged",
    "prefill_m64_bf16_flat",
    "topk",
)

_CAKE_MSA_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
}
_CAKE_MSA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_CAKE_MSA_TARGET_MINOR=0",
    "sm100f": "-DFLASHINFER_CAKE_MSA_TARGET_FAMILY=100",
}


def _get_cake_msa_csrc_dir() -> Path:
    """Locate CAKE MSA CUDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_msa"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_msa"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "CAKE MSA CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_cake_msa_include_dir() -> Path:
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


def get_cake_msa_uri(variant: CakeMSAVariant, target: CakeMSATarget) -> str:
    """Return the target-specific JIT/AOT key for one CAKE MSA variant."""

    if variant not in CAKE_MSA_VARIANTS:
        raise ValueError(f"unsupported CAKE MSA variant: {variant}")
    if target not in _CAKE_MSA_NVCC_FLAGS:
        raise ValueError(f"unsupported CAKE MSA target: {target}")
    return f"cake_msa_{variant}_{target}"


@functools.cache
def gen_cake_msa_module(variant: CakeMSAVariant, target: CakeMSATarget) -> JitSpec:
    """Generate one exact-SM100a or SM100-family CAKE MSA JIT module."""

    csrc_dir = _get_cake_msa_csrc_dir()
    include_dir = _get_cake_msa_include_dir()
    uri = get_cake_msa_uri(variant, target)
    body = csrc_dir / f"cake_msa_{variant}.cu"
    binding = csrc_dir / f"cake_msa_{variant}_binding.cu"
    if not body.exists():
        raise FileNotFoundError(f"CAKE MSA CUDA source not found: {body}")
    if not binding.exists():
        raise FileNotFoundError(f"CAKE MSA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_CAKE_MSA_NVCC_FLAGS[target],
            _CAKE_MSA_TARGET_DEFINE[target],
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            include_dir,
        ],
    )
    logger.info(f"Generated CAKE MSA {variant} {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def load_cake_msa_module(variant: CakeMSAVariant, target: CakeMSATarget):
    """Build or load one physical, target-specific CAKE MSA module."""

    module = gen_cake_msa_module(variant, target).build_and_load()
    logger.info(f"Loaded CAKE MSA {variant} {target} module")
    return module


def get_cake_msa_module(variant: CakeMSAVariant, target: CakeMSATarget):
    """Return the loaded module used by the MSA backend dispatcher."""

    return load_cake_msa_module(variant, target)


__all__ = [
    "CAKE_MSA_VARIANTS",
    "CakeMSATarget",
    "CakeMSAVariant",
    "gen_cake_msa_module",
    "get_cake_msa_module",
    "get_cake_msa_uri",
    "load_cake_msa_module",
]
