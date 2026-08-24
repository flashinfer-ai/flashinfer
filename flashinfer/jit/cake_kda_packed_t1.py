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
    sm100f_nvcc_flags,
)
from .utils import write_if_different

CakeKDAPackedT1Variant = Literal[
    "register_tile16",
    "register_tile8_interleaved",
    "register_tile16_warp",
    "cpasync_tile64_ilp4",
    "cpasync_tile64",
    "cpasync_tile128_ilp4",
    "cpasync_tile64_register_pipeline",
    "cpasync_tile128_packed_state_v_private_prefetch",
    "cpasync_tile128_v_private_prefetch",
    "cpasync_tile128_paired_row_pipeline",
    "cpasync_tile128_register_pipeline",
    "cpasync_tile128_ilp2",
]
CakeKDAPackedT1Target = Literal["sm100a", "sm100f"]

CAKE_KDA_PACKED_T1_VARIANTS: tuple[CakeKDAPackedT1Variant, ...] = (
    "register_tile16",
    "register_tile8_interleaved",
    "register_tile16_warp",
    "cpasync_tile64_ilp4",
    "cpasync_tile64",
    "cpasync_tile128_ilp4",
    "cpasync_tile64_register_pipeline",
    "cpasync_tile128_packed_state_v_private_prefetch",
    "cpasync_tile128_v_private_prefetch",
    "cpasync_tile128_paired_row_pipeline",
    "cpasync_tile128_register_pipeline",
    "cpasync_tile128_ilp2",
)

_CAKE_KDA_PACKED_T1_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
}
_CAKE_KDA_PACKED_T1_TARGET_KIND = {"sm100a": 1000, "sm100f": 100}


class CakeKDAPackedT1VariantMetadata(NamedTuple):
    body: str
    symbol: str
    value_tiles: int
    threads: int
    smem_bytes: int
    requires_aux_vec4: bool
    extra_cuda_flags: tuple[str, ...] = ("--maxrregcount=128",)


CAKE_KDA_PACKED_T1_VARIANT_METADATA: dict[
    CakeKDAPackedT1Variant, CakeKDAPackedT1VariantMetadata
] = {
    "register_tile16": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_register_tile16.cu",
        symbol="kernel_flashinfer_packed_kda_t1_register_tile16",
        value_tiles=8,
        threads=128,
        smem_bytes=0,
        requires_aux_vec4=True,
    ),
    "register_tile8_interleaved": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_register_tile8_interleaved.cu",
        symbol="kernel_flashinfer_packed_kda_t1_register_tile8_interleaved",
        value_tiles=16,
        threads=32,
        smem_bytes=0,
        requires_aux_vec4=True,
        extra_cuda_flags=("--ftz=false", "--maxrregcount=128"),
    ),
    "register_tile16_warp": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_register_tile16_warp.cu",
        symbol="kernel_flashinfer_packed_kda_t1_register_tile16_warp",
        value_tiles=8,
        threads=32,
        smem_bytes=0,
        requires_aux_vec4=True,
    ),
    "cpasync_tile64_ilp4": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile64_ilp4.cu",
        symbol="kernel_flashinfer_packed_kda_t1_cpasync_tile64_ilp4",
        value_tiles=2,
        threads=128,
        smem_bytes=24576,
        requires_aux_vec4=False,
    ),
    "cpasync_tile64": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile64.cu",
        symbol="kernel_flashinfer_packed_kda_t1_cpasync",
        value_tiles=2,
        threads=128,
        smem_bytes=16384,
        requires_aux_vec4=False,
    ),
    "cpasync_tile128_ilp4": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile128_ilp4.cu",
        symbol="kernel_flashinfer_packed_kda_t1_cpasync_tile128_ilp4",
        value_tiles=1,
        threads=128,
        smem_bytes=24576,
        requires_aux_vec4=False,
    ),
    "cpasync_tile64_register_pipeline": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile64_register_pipeline.cu",
        symbol="kernel_flashinfer_packed_kda_t1_cpasync_tile64_register_pipeline",
        value_tiles=2,
        threads=128,
        smem_bytes=16384,
        requires_aux_vec4=True,
    ),
    "cpasync_tile128_packed_state_v_private_prefetch": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile128_packed_state_v_private_prefetch.cu",
        symbol=(
            "kernel_flashinfer_packed_kda_t1_cpasync_"
            "tile128_packed_state_v_private_prefetch"
        ),
        value_tiles=1,
        threads=128,
        smem_bytes=20736,
        requires_aux_vec4=True,
    ),
    "cpasync_tile128_v_private_prefetch": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile128_v_private_prefetch.cu",
        symbol="kernel_flashinfer_packed_kda_t1_cpasync_tile128_v_private_prefetch",
        value_tiles=1,
        threads=128,
        smem_bytes=20736,
        requires_aux_vec4=True,
    ),
    "cpasync_tile128_paired_row_pipeline": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile128_paired_row_pipeline.cu",
        symbol="kernel_flashinfer_packed_kda_t1_cpasync_tile128_paired_row_pipeline",
        value_tiles=1,
        threads=128,
        smem_bytes=20736,
        requires_aux_vec4=True,
    ),
    "cpasync_tile128_register_pipeline": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile128_register_pipeline.cu",
        symbol="kernel_flashinfer_packed_kda_t1_cpasync_tile128_register_pipeline",
        value_tiles=1,
        threads=128,
        smem_bytes=20480,
        requires_aux_vec4=True,
    ),
    "cpasync_tile128_ilp2": CakeKDAPackedT1VariantMetadata(
        body="cake_kda_packed_t1_cpasync_tile128_ilp2.cu",
        symbol="kernel_flashinfer_packed_kda_t1_cpasync_tile128",
        value_tiles=1,
        threads=128,
        smem_bytes=20480,
        requires_aux_vec4=True,
    ),
}


def select_cake_kda_packed_t1_variant(
    batch: int,
    *,
    state_aligned: bool,
    aux_vec4_aligned: bool,
) -> Optional[CakeKDAPackedT1Variant]:
    """Return the qualified final selector, or ``None`` for the legacy route."""

    if batch <= 0:
        raise ValueError(f"packed KDA T=1 batch must be positive, got {batch}")
    if not state_aligned:
        return None
    if aux_vec4_aligned:
        if batch <= 14:
            return "register_tile16"
        if batch <= 29:
            return "register_tile8_interleaved"
        if batch <= 38:
            return "register_tile16_warp"
        if batch <= 41:
            return "cpasync_tile64_register_pipeline"
        if batch <= 80:
            return "cpasync_tile128_packed_state_v_private_prefetch"
        if batch <= 101:
            return "cpasync_tile128_v_private_prefetch"
        if batch <= 152:
            return "cpasync_tile128_paired_row_pipeline"
        return "cpasync_tile128_register_pipeline"
    if batch <= 24:
        return "cpasync_tile64_ilp4"
    if batch <= 37:
        return "cpasync_tile64"
    if batch == 38:
        return "cpasync_tile128_ilp4"
    return None


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "frozen Cake KDA packed T=1 sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n  - {checkout}"
    )


def get_cake_kda_packed_t1_uri(
    variant: CakeKDAPackedT1Variant,
    target: CakeKDAPackedT1Target,
) -> str:
    if variant not in CAKE_KDA_PACKED_T1_VARIANTS:
        raise ValueError(f"unsupported Cake KDA packed T=1 variant: {variant}")
    if target not in _CAKE_KDA_PACKED_T1_NVCC_FLAGS:
        raise ValueError(f"unsupported Cake KDA packed T=1 target: {target}")
    return f"cake_kda_packed_t1_{variant}_{target}"


def _get_binding_cu(metadata: CakeKDAPackedT1VariantMetadata) -> str:
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#define CAKE_KDA_PACKED_T1_BODY_FILE "{metadata.body}"
#define CAKE_KDA_PACKED_T1_KERNEL {metadata.symbol}
#define CAKE_KDA_PACKED_T1_VALUE_TILES {metadata.value_tiles}
#define CAKE_KDA_PACKED_T1_THREADS {metadata.threads}
#define CAKE_KDA_PACKED_T1_SMEM_BYTES {metadata.smem_bytes}
#define CAKE_KDA_PACKED_T1_REQUIRES_AUX_VEC4 {int(metadata.requires_aux_vec4)}

#include "cake_kda_packed_t1_binding.cuh"
"""


@functools.cache
def gen_cake_kda_packed_t1_module(
    variant: CakeKDAPackedT1Variant,
    target: CakeKDAPackedT1Target,
) -> JitSpec:
    if variant not in CAKE_KDA_PACKED_T1_VARIANTS:
        raise ValueError(f"unsupported Cake KDA packed T=1 variant: {variant}")
    if target not in _CAKE_KDA_PACKED_T1_NVCC_FLAGS:
        raise ValueError(f"unsupported Cake KDA packed T=1 target: {target}")

    csrc_dir = _get_csrc_dir()
    metadata = CAKE_KDA_PACKED_T1_VARIANT_METADATA[variant]
    body = csrc_dir / metadata.body
    if not body.exists():
        raise FileNotFoundError(f"frozen Cake KDA packed T=1 body not found: {body}")
    binding_header = csrc_dir / "cake_kda_packed_t1_binding.cuh"
    if not binding_header.exists():
        raise FileNotFoundError(
            f"Cake KDA packed T=1 binding header not found: {binding_header}"
        )

    uri = get_cake_kda_packed_t1_uri(variant, target)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_kda_packed_t1_binding.cu"
    write_if_different(binding, _get_binding_cu(metadata))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_CAKE_KDA_PACKED_T1_NVCC_FLAGS[target],
            (
                "-DFLASHINFER_CAKE_KDA_PACKED_T1_TARGET_KIND="
                f"{_CAKE_KDA_PACKED_T1_TARGET_KIND[target]}"
            ),
            *metadata.extra_cuda_flags,
        ],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
    )
    logger.info(
        "Generated Cake KDA packed T=1 %s %s JIT spec: %s", variant, target, spec.name
    )
    return spec


@functools.cache
def load_cake_kda_packed_t1_module(
    variant: CakeKDAPackedT1Variant,
    target: CakeKDAPackedT1Target,
):
    module = gen_cake_kda_packed_t1_module(variant, target).build_and_load()
    logger.info("Loaded Cake KDA packed T=1 %s %s module", variant, target)
    return module


def get_cake_kda_packed_t1_module(
    variant: CakeKDAPackedT1Variant,
    target: CakeKDAPackedT1Target,
):
    return load_cake_kda_packed_t1_module(variant, target)


__all__ = [
    "CAKE_KDA_PACKED_T1_VARIANTS",
    "CAKE_KDA_PACKED_T1_VARIANT_METADATA",
    "CakeKDAPackedT1Target",
    "CakeKDAPackedT1Variant",
    "CakeKDAPackedT1VariantMetadata",
    "gen_cake_kda_packed_t1_module",
    "get_cake_kda_packed_t1_module",
    "get_cake_kda_packed_t1_uri",
    "load_cake_kda_packed_t1_module",
    "select_cake_kda_packed_t1_variant",
]
