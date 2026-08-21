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

FlashKDAVariant = Literal[
    "m64",
    "m128",
    "m128_n16",
    "persistent_m128",
    "small_bh_m128",
]
FlashKDATarget = Literal["sm100a", "sm100f"]

FLASH_KDA_VARIANTS: tuple[FlashKDAVariant, ...] = (
    "m64",
    "m128",
    "m128_n16",
    "persistent_m128",
    "small_bh_m128",
)

_FLASH_KDA_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
}
_FLASH_KDA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
    "sm100f": "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100",
}

# Keep every frozen cache key tied to its complete generated-plus-integration
# implementation. This prevents an installed JIT/AOT cache from satisfying a
# refreshed export or binding specialization after an in-place package upgrade.
_FLASH_KDA_MODULE_IDENTS = {
    "m64": "9a5566f3be",
    "m128": "ea022a2f1f",
    "m128_n16": "ef8b47d690",
    "persistent_m128": "64bc19d01c",
    "small_bh_m128": "73369168de",
}

_FLASH_KDA_BINDING_STEMS = {
    "m64": "flashkda_bf16_fused_m64",
    "m128": "flashkda_bf16_fused_m128",
    "m128_n16": "cake_flashkda_bf16_fused_m128_n16",
    "persistent_m128": "cake_flashkda_bf16_persistent_m128",
    "small_bh_m128": "cake_flashkda_bf16_small_bh_m128",
}


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


def get_flash_kda_uri(variant: FlashKDAVariant, target: FlashKDATarget) -> str:
    """Return the target-specific JIT/AOT key for one schedule."""

    if variant not in FLASH_KDA_VARIANTS:
        raise ValueError(f"unsupported FlashKDA variant: {variant}")
    if target not in _FLASH_KDA_NVCC_FLAGS:
        raise ValueError(f"unsupported FlashKDA target: {target}")
    module_ident = _FLASH_KDA_MODULE_IDENTS[variant]
    return f"flash_kda_bf16_{variant}_{module_ident}_{target}"


@functools.cache
def gen_flash_kda_module(variant: FlashKDAVariant, target: FlashKDATarget) -> JitSpec:
    """Generate one legacy exact-SM100a or SM100-family JIT module.

    Each physical schedule is compiled in its own translation unit because the
    checked-in frozen sources intentionally retain generated helper names and
    macros. ``gen_jit_spec`` supplies FlashInfer's standard ``-use_fast_math``
    flag. CUDA 12.8 uses the exact ``sm_100a`` target on B200. CUDA 12.9 and
    newer use one ``sm_100f`` target validated on CC 10.0 and CC 10.3.
    """

    csrc_dir = _get_flash_kda_csrc_dir()
    include_dir = _get_flash_kda_include_dir()
    uri = get_flash_kda_uri(variant, target)
    binding = csrc_dir / f"{_FLASH_KDA_BINDING_STEMS[variant]}_binding.cu"
    if not binding.exists():
        raise FileNotFoundError(f"FlashKDA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_FLASH_KDA_NVCC_FLAGS[target],
            _FLASH_KDA_TARGET_DEFINE[target],
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            include_dir,
        ],
    )
    logger.info(f"Generated FlashKDA {variant} {target} JIT spec: {spec.name}")
    return spec


def gen_flash_kda_m64_module(target: FlashKDATarget) -> JitSpec:
    """Generate the fixed N=1, H=64 two-CTA M64 module."""

    return gen_flash_kda_module("m64", target)


def gen_flash_kda_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the general packed/fixed M128 module."""

    return gen_flash_kda_module("m128", target)


def gen_flash_kda_m128_n16_module(target: FlashKDATarget) -> JitSpec:
    """Generate the H12 packed/fixed M128 module with a 16-token chunk."""

    return gen_flash_kda_module("m128_n16", target)


def gen_flash_kda_persistent_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the SM100-only static-binned persistent M128 module."""

    return gen_flash_kda_module("persistent_m128", target)


def gen_flash_kda_small_bh_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the fixed-layout small-BH owner/helper M128 module."""

    return gen_flash_kda_module("small_bh_m128", target)


@functools.cache
def load_flash_kda_module(variant: FlashKDAVariant, target: FlashKDATarget):
    """Build or load one physical, target-specific FlashKDA module."""

    module = gen_flash_kda_module(variant, target).build_and_load()
    logger.info(f"Loaded FlashKDA {variant} {target} module")
    return module


def load_flash_kda_m64_module(target: FlashKDATarget):
    """Load the fixed N=1, H=64 two-CTA M64 module."""

    return load_flash_kda_module("m64", target)


def load_flash_kda_m128_module(target: FlashKDATarget):
    """Load the general packed/fixed M128 module."""

    return load_flash_kda_module("m128", target)


def load_flash_kda_m128_n16_module(target: FlashKDATarget):
    """Load the H12 packed/fixed M128 module with a 16-token chunk."""

    return load_flash_kda_module("m128_n16", target)


def load_flash_kda_persistent_m128_module(target: FlashKDATarget):
    """Load the SM100-only static-binned persistent M128 module."""

    return load_flash_kda_module("persistent_m128", target)


def load_flash_kda_small_bh_m128_module(target: FlashKDATarget):
    """Load the fixed-layout small-BH owner/helper M128 module."""

    return load_flash_kda_module("small_bh_m128", target)


def get_flash_kda_prefill_module(variant: FlashKDAVariant, target: FlashKDATarget):
    """Return the loaded module used by the recurrent-KDA prefill dispatcher."""

    return load_flash_kda_module(variant, target)


__all__ = [
    "FLASH_KDA_VARIANTS",
    "FlashKDATarget",
    "FlashKDAVariant",
    "gen_flash_kda_m64_module",
    "gen_flash_kda_m128_module",
    "gen_flash_kda_m128_n16_module",
    "gen_flash_kda_persistent_m128_module",
    "gen_flash_kda_small_bh_m128_module",
    "gen_flash_kda_module",
    "get_flash_kda_prefill_module",
    "get_flash_kda_uri",
    "load_flash_kda_m64_module",
    "load_flash_kda_m128_module",
    "load_flash_kda_m128_n16_module",
    "load_flash_kda_persistent_m128_module",
    "load_flash_kda_small_bh_m128_module",
    "load_flash_kda_module",
]
