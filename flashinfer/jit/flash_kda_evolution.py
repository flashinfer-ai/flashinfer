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
import hashlib
from pathlib import Path
from typing import Literal, NamedTuple

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
)
from .utils import write_if_different

FlashKDAEvolutionVariant = Literal[
    "m128_h32_p0_s1",
    "m128_h4_p0_s1",
    "m128_h64_p1_s114",
    "m128_h64_p1_s126",
    "m128_h96_p0_s1",
    "m128_h96_p1_s166",
    "m128_h96_p1_s173",
    "m64_f1_t8192_h64",
    "vtile_f0_t16_h96_p1_s192",
    "vtile_f0_t16_h96_p1_s96",
    "vtile_f0_t37_h96_p1_s96",
    "vtile_f0_t97_h96_p1_s96",
    "vtile_f1_t1024_h64_p4_s128",
    "vtile_f1_t1024_h96_p1_s12288",
    "vtile_f1_t1024_h96_p1_s1536",
    "vtile_f1_t1024_h96_p1_s24576",
    "vtile_f1_t1024_h96_p1_s3072",
    "vtile_f1_t1024_h96_p1_s6144",
    "vtile_f1_t1024_h96_p6_s128",
    "vtile_f1_t1048576_h1_p1_s1",
    "vtile_f1_t131072_h1_p1_s1",
    "vtile_f1_t16384_h16_p1_s16",
    "vtile_f1_t32768_h16_p1_s16",
    "vtile_f1_t524288_h1_p1_s2",
    "vtile_f1_t65536_h16_p1_s16",
    "vtile_f1_t65536_h4_p1_s4",
    "vtile_f1_t65536_h8_p1_s8",
    "vtile_f1_t8192_h32_p1_s32",
    "vtile_f1_t8192_h96_p1_s96",
]
FlashKDAEvolutionTarget = Literal["sm100a", "sm100f"]


class FlashKDAEvolutionMetadata(NamedTuple):
    source_stem: str
    kernel_symbol: str
    value_rows: int
    has_tile_schedule: bool


_FLASH_KDA_EVOLUTION_VARIANT_NAMES: tuple[FlashKDAEvolutionVariant, ...] = (
    "m128_h32_p0_s1",
    "m128_h4_p0_s1",
    "m128_h64_p1_s114",
    "m128_h64_p1_s126",
    "m128_h96_p0_s1",
    "m128_h96_p1_s166",
    "m128_h96_p1_s173",
    "m64_f1_t8192_h64",
    "vtile_f0_t16_h96_p1_s192",
    "vtile_f0_t16_h96_p1_s96",
    "vtile_f0_t37_h96_p1_s96",
    "vtile_f0_t97_h96_p1_s96",
    "vtile_f1_t1024_h64_p4_s128",
    "vtile_f1_t1024_h96_p1_s12288",
    "vtile_f1_t1024_h96_p1_s1536",
    "vtile_f1_t1024_h96_p1_s24576",
    "vtile_f1_t1024_h96_p1_s3072",
    "vtile_f1_t1024_h96_p1_s6144",
    "vtile_f1_t1024_h96_p6_s128",
    "vtile_f1_t1048576_h1_p1_s1",
    "vtile_f1_t131072_h1_p1_s1",
    "vtile_f1_t16384_h16_p1_s16",
    "vtile_f1_t32768_h16_p1_s16",
    "vtile_f1_t524288_h1_p1_s2",
    "vtile_f1_t65536_h16_p1_s16",
    "vtile_f1_t65536_h4_p1_s4",
    "vtile_f1_t65536_h8_p1_s8",
    "vtile_f1_t8192_h32_p1_s32",
    "vtile_f1_t8192_h96_p1_s96",
)
_TILE_SCHEDULE_VARIANTS = frozenset(
    {
        "m128_h32_p0_s1",
        "m128_h4_p0_s1",
        "m128_h64_p1_s114",
        "m128_h64_p1_s126",
        "m128_h96_p0_s1",
        "m128_h96_p1_s166",
        "m128_h96_p1_s173",
    }
)
_VALUE_ROWS_64_VARIANTS = frozenset({"m64_f1_t8192_h64"})


FLASH_KDA_EVOLUTION_VARIANTS: dict[
    FlashKDAEvolutionVariant, FlashKDAEvolutionMetadata
] = {
    variant: FlashKDAEvolutionMetadata(
        source_stem=f"cake_flashkda_blackwell_evolution_{variant}",
        kernel_symbol=f"kernel_flashkda_blackwell_evolution_{variant}",
        value_rows=64 if variant in _VALUE_ROWS_64_VARIANTS else 128,
        has_tile_schedule=variant in _TILE_SCHEDULE_VARIANTS,
    )
    for variant in _FLASH_KDA_EVOLUTION_VARIANT_NAMES
}

_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
}
_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
    "sm100f": "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100",
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashKDA Blackwell evolution sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _binding_source(metadata: FlashKDAEvolutionMetadata) -> str:
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

#define FLASHKDA_BLACKWELL_EVOLUTION_BODY_FILE "{metadata.source_stem}.cu"
#define FLASHKDA_BLACKWELL_EVOLUTION_KERNEL {metadata.kernel_symbol}
#define FLASHKDA_BLACKWELL_EVOLUTION_VALUE_ROWS {metadata.value_rows}
#define FLASHKDA_BLACKWELL_EVOLUTION_HAS_TILE_SCHEDULE {int(metadata.has_tile_schedule)}

#include "cake_flashkda_blackwell_evolution_binding.cuh"
"""


def _module_ident(csrc_dir: Path, metadata: FlashKDAEvolutionMetadata) -> str:
    digest = hashlib.sha256()
    paths = (
        csrc_dir / f"{metadata.source_stem}.cu",
        csrc_dir / "cake_flashkda_blackwell_evolution_binding.cuh",
        csrc_dir / "flashkda_binding_common.cuh",
    )
    for index, path in enumerate(paths):
        if index:
            digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def get_flash_kda_evolution_uri(
    variant: FlashKDAEvolutionVariant, target: FlashKDAEvolutionTarget
) -> str:
    if variant not in FLASH_KDA_EVOLUTION_VARIANTS:
        raise ValueError(f"unsupported FlashKDA evolution variant: {variant}")
    if target not in _NVCC_FLAGS:
        raise ValueError(f"unsupported FlashKDA evolution target: {target}")
    csrc_dir = _get_csrc_dir()
    ident = _module_ident(csrc_dir, FLASH_KDA_EVOLUTION_VARIANTS[variant])
    return f"flash_kda_evolution_{variant}_{ident}_{target}"


@functools.cache
def gen_flash_kda_evolution_module(
    variant: FlashKDAEvolutionVariant, target: FlashKDAEvolutionTarget
) -> JitSpec:
    if variant not in FLASH_KDA_EVOLUTION_VARIANTS:
        raise ValueError(f"unsupported FlashKDA evolution variant: {variant}")
    if target not in _NVCC_FLAGS:
        raise ValueError(f"unsupported FlashKDA evolution target: {target}")
    csrc_dir = _get_csrc_dir()
    metadata = FLASH_KDA_EVOLUTION_VARIANTS[variant]
    body = csrc_dir / f"{metadata.source_stem}.cu"
    binding_header = csrc_dir / "cake_flashkda_blackwell_evolution_binding.cuh"
    common_header = csrc_dir / "flashkda_binding_common.cuh"
    for path in (body, binding_header, common_header):
        if not path.exists():
            raise FileNotFoundError(f"FlashKDA evolution source not found: {path}")
    uri = get_flash_kda_evolution_uri(variant, target)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "binding.cu"
    write_if_different(binding, _binding_source(metadata))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            "--use_fast_math",
            *_NVCC_FLAGS[target],
            _TARGET_DEFINE[target],
        ],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
    )
    logger.info(
        "Generated FlashKDA evolution %s %s JIT spec: %s", variant, target, spec.name
    )
    return spec


@functools.cache
def load_flash_kda_evolution_module(
    variant: FlashKDAEvolutionVariant, target: FlashKDAEvolutionTarget
):
    module = gen_flash_kda_evolution_module(variant, target).build_and_load()
    logger.info("Loaded FlashKDA evolution %s %s module", variant, target)
    return module


__all__ = [
    "FLASH_KDA_EVOLUTION_VARIANTS",
    "FlashKDAEvolutionMetadata",
    "FlashKDAEvolutionTarget",
    "FlashKDAEvolutionVariant",
    "gen_flash_kda_evolution_module",
    "get_flash_kda_evolution_uri",
    "load_flash_kda_evolution_module",
]
