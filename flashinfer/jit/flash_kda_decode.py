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

FlashKDADecodeVariant = Literal[
    "d128_t5_precomputed_gram_split1",
    "d128_t5_precomputed_gram_split2",
    "d128_t5_precomputed_gram_split4",
    "d128_t5_precomputed_gram_split8",
]

FLASH_KDA_DECODE_VARIANTS: tuple[FlashKDADecodeVariant, ...] = (
    "d128_t5_precomputed_gram_split1",
    "d128_t5_precomputed_gram_split2",
    "d128_t5_precomputed_gram_split4",
    "d128_t5_precomputed_gram_split8",
)


def _get_csrc_dir() -> Path:
    """Locate the frozen decode sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "frozen FlashKDA decode sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
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


def get_flash_kda_decode_uri(variant: FlashKDADecodeVariant) -> str:
    """Return the stable JIT/AOT cache key for one physical decode schedule."""

    if variant not in FLASH_KDA_DECODE_VARIANTS:
        raise ValueError(f"unsupported FlashKDA decode variant: {variant}")
    return f"flash_kda_decode_{variant}_sm100a"


@functools.cache
def gen_flash_kda_decode_module(variant: FlashKDADecodeVariant) -> JitSpec:
    """Generate one exact-sm_100a frozen FlashKDA decode module."""

    csrc_dir = _get_csrc_dir()
    binding = csrc_dir / f"flashkda_decode_{variant}_binding.cu"
    if not binding.exists():
        raise FileNotFoundError(
            f"frozen FlashKDA decode binding source not found: {binding}"
        )

    spec = gen_jit_spec(
        name=get_flash_kda_decode_uri(variant),
        sources=[binding],
        extra_cuda_cflags=[*sm100a_nvcc_flags, "--maxrregcount=128"],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_include_dir(),
        ],
    )
    logger.info(f"Generated FlashKDA decode {variant} JIT spec: {spec.name}")
    return spec


@functools.cache
def load_flash_kda_decode_module(variant: FlashKDADecodeVariant):
    """Build or load one physical FlashKDA decode module."""

    module = gen_flash_kda_decode_module(variant).build_and_load()
    logger.info(f"Loaded FlashKDA decode {variant} module")
    return module


def get_flash_kda_decode_module(variant: FlashKDADecodeVariant):
    """Return the loaded module used by the recurrent-KDA dispatcher."""

    return load_flash_kda_decode_module(variant)


__all__ = [
    "FLASH_KDA_DECODE_VARIANTS",
    "FlashKDADecodeVariant",
    "gen_flash_kda_decode_module",
    "get_flash_kda_decode_module",
    "get_flash_kda_decode_uri",
    "load_flash_kda_decode_module",
]
