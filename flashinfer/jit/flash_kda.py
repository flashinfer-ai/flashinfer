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
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags

FlashKDAVariant = Literal["m64", "m128"]


def _get_flash_kda_csrc_dir() -> Path:
    """Locate frozen FlashKDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "FlashKDA CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_flash_kda_include_dir() -> Path:
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


def get_flash_kda_uri(variant: FlashKDAVariant) -> str:
    """Return the stable JIT/AOT cache key for one physical schedule."""

    if variant not in ("m64", "m128"):
        raise ValueError(f"unsupported FlashKDA variant: {variant}")
    return f"flash_kda_bf16_fused_{variant}_sm100a"


@functools.cache
def gen_flash_kda_module(variant: FlashKDAVariant) -> JitSpec:
    """Generate one exact-sm_100a FlashKDA JIT module.

    Each physical schedule is compiled in its own translation unit because the
    checked-in frozen sources intentionally retain generated helper names and
    macros. ``gen_jit_spec`` supplies FlashInfer's standard ``-use_fast_math``
    flag; the explicit architecture flags below emit only an sm_100a cubin.
    """

    csrc_dir = _get_flash_kda_csrc_dir()
    include_dir = _get_flash_kda_include_dir()
    uri = get_flash_kda_uri(variant)
    binding = csrc_dir / f"flashkda_bf16_fused_{variant}_binding.cu"
    if not binding.exists():
        raise FileNotFoundError(f"FlashKDA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=sm100a_nvcc_flags,
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            include_dir,
        ],
    )
    logger.info(f"Generated FlashKDA {variant} JIT spec: {spec.name}")
    return spec


def gen_flash_kda_m64_module() -> JitSpec:
    """Generate the fixed N=1, H=64 two-CTA M64 module."""

    return gen_flash_kda_module("m64")


def gen_flash_kda_m128_module() -> JitSpec:
    """Generate the general packed/fixed M128 module."""

    return gen_flash_kda_module("m128")


@functools.cache
def load_flash_kda_module(variant: FlashKDAVariant):
    """Build or load one physical FlashKDA module."""

    module = gen_flash_kda_module(variant).build_and_load()
    logger.info(f"Loaded FlashKDA {variant} module")
    return module


def load_flash_kda_m64_module():
    """Load the fixed N=1, H=64 two-CTA M64 module."""

    return load_flash_kda_module("m64")


def load_flash_kda_m128_module():
    """Load the general packed/fixed M128 module."""

    return load_flash_kda_module("m128")


def get_flash_kda_prefill_module(variant: FlashKDAVariant):
    """Return the loaded module used by the recurrent-KDA prefill dispatcher."""

    return load_flash_kda_module(variant)


__all__ = [
    "FlashKDAVariant",
    "gen_flash_kda_m64_module",
    "gen_flash_kda_m128_module",
    "gen_flash_kda_module",
    "get_flash_kda_prefill_module",
    "get_flash_kda_uri",
    "load_flash_kda_m64_module",
    "load_flash_kda_m128_module",
    "load_flash_kda_module",
]
