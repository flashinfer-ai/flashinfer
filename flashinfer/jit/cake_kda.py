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

CakeKDAVariant = Literal[
    "m128_unbounded_softplus",
    "m128_bt64_unbounded_softplus",
]
CakeKDATarget = Literal["sm100a", "sm103a"]

CAKE_KDA_VARIANTS: tuple[CakeKDAVariant, ...] = (
    "m128_unbounded_softplus",
    "m128_bt64_unbounded_softplus",
)

_CAKE_KDA_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_CAKE_KDA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=3",
}

# Keep the frozen cache key tied to the complete generated-plus-integration
# implementation so an installed cache cannot satisfy a refreshed export.
_CAKE_KDA_MODULE_IDENTS = {
    "m128_unbounded_softplus": "d7a7b33c69",
    "m128_bt64_unbounded_softplus": "8f5147c17f",
}


def _get_cake_kda_csrc_dir() -> Path:
    """Locate frozen CakeKDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "CakeKDA CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_cake_kda_include_dir() -> Path:
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


def get_cake_kda_uri(variant: CakeKDAVariant, target: CakeKDATarget) -> str:
    """Return the target-specific JIT/AOT key for one schedule."""

    if variant not in CAKE_KDA_VARIANTS:
        raise ValueError(f"unsupported CakeKDA variant: {variant}")
    if target not in _CAKE_KDA_NVCC_FLAGS:
        raise ValueError(f"unsupported CakeKDA target: {target}")
    module_ident = _CAKE_KDA_MODULE_IDENTS[variant]
    return f"cake_kda_bf16_fused_{variant}_{module_ident}_{target}"


@functools.cache
def gen_cake_kda_module(variant: CakeKDAVariant, target: CakeKDATarget) -> JitSpec:
    """Generate one exact-SM100a or exact-SM103a JIT module.

    Each physical schedule is compiled in its own translation unit because the
    checked-in frozen sources intentionally retain generated helper names and
    macros. ``gen_jit_spec`` supplies FlashInfer's standard ``-use_fast_math``
    flag. B200 and B300 use separate exact targets and therefore separate
    cubins and cache identities.
    """

    csrc_dir = _get_cake_kda_csrc_dir()
    include_dir = _get_cake_kda_include_dir()
    uri = get_cake_kda_uri(variant, target)
    binding = csrc_dir / f"cake_kda_bf16_fused_{variant}_binding.cu"
    if not binding.exists():
        raise FileNotFoundError(f"CakeKDA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_CAKE_KDA_NVCC_FLAGS[target],
            _CAKE_KDA_TARGET_DEFINE[target],
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            include_dir,
        ],
    )
    logger.info(f"Generated CakeKDA {variant} {target} JIT spec: {spec.name}")
    return spec


def gen_cake_kda_m128_unbounded_softplus_module(target: CakeKDATarget) -> JitSpec:
    """Generate the native unbounded-softplus M128 module."""

    return gen_cake_kda_module("m128_unbounded_softplus", target)


def gen_cake_kda_m128_bt64_unbounded_softplus_module(
    target: CakeKDATarget,
) -> JitSpec:
    """Generate the checkpoint-aligned native unbounded-softplus BT64 module."""

    return gen_cake_kda_module("m128_bt64_unbounded_softplus", target)


@functools.cache
def load_cake_kda_module(variant: CakeKDAVariant, target: CakeKDATarget):
    """Build or load one physical, target-specific CakeKDA module."""

    module = gen_cake_kda_module(variant, target).build_and_load()
    logger.info(f"Loaded CakeKDA {variant} {target} module")
    return module


def load_cake_kda_m128_unbounded_softplus_module(target: CakeKDATarget):
    """Load the native unbounded-softplus M128 module."""

    return load_cake_kda_module("m128_unbounded_softplus", target)


def load_cake_kda_m128_bt64_unbounded_softplus_module(target: CakeKDATarget):
    """Load the checkpoint-aligned native unbounded-softplus BT64 module."""

    return load_cake_kda_module("m128_bt64_unbounded_softplus", target)


def get_cake_kda_prefill_module(variant: CakeKDAVariant, target: CakeKDATarget):
    """Return the loaded module used by the recurrent-KDA prefill dispatcher."""

    return load_cake_kda_module(variant, target)


__all__ = [
    "CAKE_KDA_VARIANTS",
    "CakeKDATarget",
    "CakeKDAVariant",
    "gen_cake_kda_m128_bt64_unbounded_softplus_module",
    "gen_cake_kda_m128_unbounded_softplus_module",
    "gen_cake_kda_module",
    "get_cake_kda_prefill_module",
    "get_cake_kda_uri",
    "load_cake_kda_m128_bt64_unbounded_softplus_module",
    "load_cake_kda_m128_unbounded_softplus_module",
    "load_cake_kda_module",
]
