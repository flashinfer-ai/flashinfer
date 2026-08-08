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
from typing import Literal, NamedTuple, Optional

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from .utils import write_if_different

FlashKDAPackedT1Variant = Literal["tile8", "tile16"]
FlashKDAPackedT1Target = Literal["sm100a", "sm103a"]

FLASH_KDA_PACKED_T1_VARIANTS: tuple[FlashKDAPackedT1Variant, ...] = (
    "tile8",
    "tile16",
)

_FLASH_KDA_PACKED_T1_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

_FLASH_KDA_PACKED_T1_TARGET_KIND = {
    "sm100a": 1000,
    "sm103a": 1003,
}


class FlashKDAPackedT1VariantMetadata(NamedTuple):
    value_splits: int
    symbol: str
    batch_min: int
    batch_max: Optional[int]


FLASH_KDA_PACKED_T1_VARIANT_METADATA: dict[
    FlashKDAPackedT1Variant, FlashKDAPackedT1VariantMetadata
] = {
    "tile8": FlashKDAPackedT1VariantMetadata(
        value_splits=16,
        symbol="kernel_kimi_k3_kda_t1_packed",
        batch_min=1,
        batch_max=31,
    ),
    "tile16": FlashKDAPackedT1VariantMetadata(
        value_splits=8,
        symbol="kernel_kimi_k3_kda_t1_packed_tile16",
        batch_min=32,
        batch_max=None,
    ),
}


def _variant_for_batch(batch: int) -> FlashKDAPackedT1Variant:
    """Select the frozen schedule using only host-visible shape metadata."""

    if batch <= 0:
        raise ValueError(f"packed KDA T=1 batch must be positive, got {batch}")
    return "tile16" if batch >= 32 else "tile8"


def _get_csrc_dir() -> Path:
    """Locate the frozen packed-KDA sources in installs and checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "frozen packed KDA T=1 sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    """Locate FlashInfer headers in installs and source checkouts."""

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


def get_flash_kda_packed_t1_uri(
    variant: FlashKDAPackedT1Variant,
    target: FlashKDAPackedT1Target,
) -> str:
    """Return the exact-target JIT/AOT key for one packed schedule."""

    if variant not in FLASH_KDA_PACKED_T1_VARIANTS:
        raise ValueError(f"unsupported packed KDA T=1 variant: {variant}")
    if target not in _FLASH_KDA_PACKED_T1_NVCC_FLAGS:
        raise ValueError(f"unsupported packed KDA T=1 target: {target}")
    return f"flash_kda_packed_t1_{variant}_{target}"


def _get_binding_cu(
    variant: FlashKDAPackedT1Variant,
    metadata: FlashKDAPackedT1VariantMetadata,
) -> str:
    """Render the binding translation unit without changing the frozen body."""

    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define FLASHKDA_PACKED_T1_BODY_FILE "flashkda_packed_t1_{variant}.cu"
#define FLASHKDA_PACKED_T1_KERNEL {metadata.symbol}
#define FLASHKDA_PACKED_T1_VALUE_SPLITS {metadata.value_splits}

#include "flashkda_packed_t1_binding.cuh"
"""


@functools.cache
def gen_flash_kda_packed_t1_module(
    variant: FlashKDAPackedT1Variant,
    target: FlashKDAPackedT1Target,
) -> JitSpec:
    """Generate one exact-SM100a or exact-SM103a packed-KDA module."""

    if variant not in FLASH_KDA_PACKED_T1_VARIANTS:
        raise ValueError(f"unsupported packed KDA T=1 variant: {variant}")
    if target not in _FLASH_KDA_PACKED_T1_NVCC_FLAGS:
        raise ValueError(f"unsupported packed KDA T=1 target: {target}")

    csrc_dir = _get_csrc_dir()
    body = csrc_dir / f"flashkda_packed_t1_{variant}.cu"
    if not body.exists():
        raise FileNotFoundError(f"frozen packed KDA T=1 body not found: {body}")
    binding_header = csrc_dir / "flashkda_packed_t1_binding.cuh"
    if not binding_header.exists():
        raise FileNotFoundError(
            f"packed KDA T=1 binding header not found: {binding_header}"
        )

    metadata = FLASH_KDA_PACKED_T1_VARIANT_METADATA[variant]
    uri = get_flash_kda_packed_t1_uri(variant, target)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "flashkda_packed_t1_binding.cu"
    write_if_different(binding, _get_binding_cu(variant, metadata))

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_FLASH_KDA_PACKED_T1_NVCC_FLAGS[target],
            (
                "-DFLASHINFER_FLASH_KDA_PACKED_T1_TARGET_KIND="
                f"{_FLASH_KDA_PACKED_T1_TARGET_KIND[target]}"
            ),
            "--maxrregcount=128",
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_include_dir(),
        ],
    )
    logger.info(
        "Generated packed KDA T=1 %s %s JIT spec: %s",
        variant,
        target,
        spec.name,
    )
    return spec


@functools.cache
def load_flash_kda_packed_t1_module(
    variant: FlashKDAPackedT1Variant,
    target: FlashKDAPackedT1Target,
):
    """Build or load one exact-target packed-KDA module."""

    module = gen_flash_kda_packed_t1_module(variant, target).build_and_load()
    logger.info("Loaded packed KDA T=1 %s %s module", variant, target)
    return module


def get_flash_kda_packed_t1_module(
    variant: FlashKDAPackedT1Variant,
    target: FlashKDAPackedT1Target,
):
    """Return the loaded module used by the packed-KDA dispatcher."""

    return load_flash_kda_packed_t1_module(variant, target)


__all__ = [
    "FLASH_KDA_PACKED_T1_VARIANTS",
    "FLASH_KDA_PACKED_T1_VARIANT_METADATA",
    "FlashKDAPackedT1Target",
    "FlashKDAPackedT1Variant",
    "FlashKDAPackedT1VariantMetadata",
    "gen_flash_kda_packed_t1_module",
    "get_flash_kda_packed_t1_module",
    "get_flash_kda_packed_t1_uri",
    "load_flash_kda_packed_t1_module",
]
