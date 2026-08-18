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
from typing import Literal, NamedTuple

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
    sm103a_nvcc_flags,
)
from .utils import write_if_different

FlashKDADecodeVariant = Literal[
    "d128_t1_precomputed_split1",
    "d128_t1_precomputed_split2",
    "d128_t1_precomputed_split4",
    "d128_t1_precomputed_split8",
    "d128_t1_precomputed_direct_split16",
    "d128_t1_precomputed_direct_split8",
    "d128_t2_precomputed_split1",
    "d128_t2_precomputed_split2",
    "d128_t2_precomputed_split4",
    "d128_t2_precomputed_split8",
    "d128_t3_lower_bound_split4",
    "d128_t4_precomputed_split1",
    "d128_t4_precomputed_split2",
    "d128_t4_precomputed_split4",
    "d128_t4_precomputed_split8",
    "d128_t5_precomputed_gram_split1",
    "d128_t5_precomputed_gram_split2",
    "d128_t5_precomputed_gram_split4",
    "d128_t5_precomputed_gram_split8",
    "d128_t6_precomputed_gram_split1",
    "d128_t6_precomputed_gram_split2",
    "d128_t6_precomputed_gram_split4",
    "d128_t6_precomputed_gram_split8",
]
FlashKDADecodeTarget = Literal["sm100a", "sm100f", "sm103a"]

_FLASH_KDA_DECODE_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

# The binding uses one numeric target kind to enforce the same execution
# boundary for JIT and AOT modules. 100 is the CC 10.0/10.3 family target;
# 1000 and 1003 are exact CC targets used by the CUDA-12.8 compatibility path
# and the two GB300 direct-T1 performance specializations, respectively.
_FLASH_KDA_DECODE_TARGET_KIND = {
    "sm100f": 100,
    "sm100a": 1000,
    "sm103a": 1003,
}
FLASH_KDA_DECODE_VARIANTS: tuple[FlashKDADecodeVariant, ...] = (
    "d128_t1_precomputed_split1",
    "d128_t1_precomputed_split2",
    "d128_t1_precomputed_split4",
    "d128_t1_precomputed_split8",
    "d128_t1_precomputed_direct_split16",
    "d128_t1_precomputed_direct_split8",
    "d128_t2_precomputed_split1",
    "d128_t2_precomputed_split2",
    "d128_t2_precomputed_split4",
    "d128_t2_precomputed_split8",
    "d128_t3_lower_bound_split4",
    "d128_t4_precomputed_split1",
    "d128_t4_precomputed_split2",
    "d128_t4_precomputed_split4",
    "d128_t4_precomputed_split8",
    "d128_t5_precomputed_gram_split1",
    "d128_t5_precomputed_gram_split2",
    "d128_t5_precomputed_gram_split4",
    "d128_t5_precomputed_gram_split8",
    "d128_t6_precomputed_gram_split1",
    "d128_t6_precomputed_gram_split2",
    "d128_t6_precomputed_gram_split4",
    "d128_t6_precomputed_gram_split8",
)

FLASH_KDA_DECODE_DIRECT_VARIANTS: tuple[FlashKDADecodeVariant, ...] = (
    "d128_t1_precomputed_direct_split16",
    "d128_t1_precomputed_direct_split8",
)


class FlashKDADecodeVariantMetadata(NamedTuple):
    head_dim: int
    tokens: int
    gate_kind: int
    value_split: int
    launch_threads: int
    direct_impl: bool


def _variant_metadata(
    tokens: int,
    gate_kind: int,
    value_split: int,
    coefficient_gram: bool = False,
    direct_impl: bool = False,
) -> FlashKDADecodeVariantMetadata:
    """Derive the exact launch geometry used by the frozen Loom schedule."""

    head_dim = 128
    value_rows = head_dim // value_split
    value_warps = value_rows // 16
    rows_per_group = 2 if coefficient_gram and value_split == 8 else 8
    state_warps = (value_rows // rows_per_group + 1) // 2
    launch_threads = max(tokens, value_warps, state_warps) * 32
    return FlashKDADecodeVariantMetadata(
        head_dim,
        tokens,
        gate_kind,
        value_split,
        launch_threads,
        direct_impl,
    )


FLASH_KDA_DECODE_VARIANT_METADATA: dict[
    FlashKDADecodeVariant, FlashKDADecodeVariantMetadata
] = {
    "d128_t1_precomputed_split1": _variant_metadata(1, 0, 1),
    "d128_t1_precomputed_split2": _variant_metadata(1, 0, 2),
    "d128_t1_precomputed_split4": _variant_metadata(1, 0, 4),
    "d128_t1_precomputed_split8": _variant_metadata(1, 0, 8),
    "d128_t1_precomputed_direct_split16": _variant_metadata(1, 0, 16, direct_impl=True),
    "d128_t1_precomputed_direct_split8": _variant_metadata(1, 0, 8, direct_impl=True),
    "d128_t2_precomputed_split1": _variant_metadata(2, 0, 1),
    "d128_t2_precomputed_split2": _variant_metadata(2, 0, 2),
    "d128_t2_precomputed_split4": _variant_metadata(2, 0, 4),
    "d128_t2_precomputed_split8": _variant_metadata(2, 0, 8),
    "d128_t3_lower_bound_split4": _variant_metadata(3, 1, 4),
    "d128_t4_precomputed_split1": _variant_metadata(4, 0, 1),
    "d128_t4_precomputed_split2": _variant_metadata(4, 0, 2),
    "d128_t4_precomputed_split4": _variant_metadata(4, 0, 4),
    "d128_t4_precomputed_split8": _variant_metadata(4, 0, 8),
    "d128_t5_precomputed_gram_split1": _variant_metadata(5, 0, 1, True),
    "d128_t5_precomputed_gram_split2": _variant_metadata(5, 0, 2, True),
    "d128_t5_precomputed_gram_split4": _variant_metadata(5, 0, 4, True),
    "d128_t5_precomputed_gram_split8": _variant_metadata(5, 0, 8, True),
    "d128_t6_precomputed_gram_split1": _variant_metadata(6, 0, 1, True),
    "d128_t6_precomputed_gram_split2": _variant_metadata(6, 0, 2, True),
    "d128_t6_precomputed_gram_split4": _variant_metadata(6, 0, 4, True),
    "d128_t6_precomputed_gram_split8": _variant_metadata(6, 0, 8, True),
}


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


def get_flash_kda_decode_uri(
    variant: FlashKDADecodeVariant, target: FlashKDADecodeTarget
) -> str:
    """Return the physical-target JIT/AOT key for one decode schedule."""

    if variant not in FLASH_KDA_DECODE_VARIANTS:
        raise ValueError(f"unsupported FlashKDA decode variant: {variant}")
    if target not in _FLASH_KDA_DECODE_NVCC_FLAGS:
        raise ValueError(f"unsupported FlashKDA decode target: {target}")
    if target == "sm103a" and variant not in FLASH_KDA_DECODE_DIRECT_VARIANTS:
        raise ValueError(
            "exact SM103a FlashKDA decode modules are only retained for "
            f"direct T=1 variants, got {variant}"
        )
    return f"flash_kda_decode_{variant}_{target}"


def _get_binding_cu(
    variant: FlashKDADecodeVariant,
    metadata: FlashKDADecodeVariantMetadata,
) -> str:
    """Render the generic binding translation unit for one frozen body."""

    body_file = f"flashkda_decode_{variant}.cu"
    direct_impl_define = (
        "#define FLASHKDA_DECODE_DIRECT_IMPL 1\n" if metadata.direct_impl else ""
    )
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

#define FLASHKDA_DECODE_BODY_FILE "{body_file}"
#define FLASHKDA_DECODE_HEAD_DIM {metadata.head_dim}
#define FLASHKDA_DECODE_TOKENS {metadata.tokens}
#define FLASHKDA_DECODE_GATE_KIND {metadata.gate_kind}
#define FLASHKDA_DECODE_VALUE_SPLIT {metadata.value_split}
#define FLASHKDA_DECODE_LAUNCH_THREADS {metadata.launch_threads}
{direct_impl_define}

#include "flashkda_decode_binding.cuh"
"""


@functools.cache
def gen_flash_kda_decode_module(
    variant: FlashKDADecodeVariant, target: FlashKDADecodeTarget
) -> JitSpec:
    """Generate one family or exact-target frozen decode module.

    ``sm100f`` is the normal CUDA-12.9+ target for both CC 10.0 and CC 10.3.
    ``sm100a`` preserves CUDA-12.8 B200 support, while ``sm103a`` is limited to
    the two direct T=1 bodies whose family-target code showed a measurable
    GB300 latency regression.
    """

    csrc_dir = _get_csrc_dir()
    body = csrc_dir / f"flashkda_decode_{variant}.cu"
    if not body.exists():
        raise FileNotFoundError(f"frozen FlashKDA decode body source not found: {body}")
    binding_header = csrc_dir / "flashkda_decode_binding.cuh"
    if not binding_header.exists():
        raise FileNotFoundError(
            f"generic FlashKDA decode binding header not found: {binding_header}"
        )

    metadata = FLASH_KDA_DECODE_VARIANT_METADATA[variant]
    uri = get_flash_kda_decode_uri(variant, target)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "flashkda_decode_binding.cu"
    write_if_different(binding, _get_binding_cu(variant, metadata))

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_FLASH_KDA_DECODE_NVCC_FLAGS[target],
            (
                "-DFLASHINFER_FLASH_KDA_DECODE_TARGET_KIND="
                f"{_FLASH_KDA_DECODE_TARGET_KIND[target]}"
            ),
            "--maxrregcount=128",
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_include_dir(),
        ],
    )
    logger.info(f"Generated FlashKDA decode {variant} {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def load_flash_kda_decode_module(
    variant: FlashKDADecodeVariant, target: FlashKDADecodeTarget
):
    """Build or load one physical-target decode module."""

    module = gen_flash_kda_decode_module(variant, target).build_and_load()
    logger.info(f"Loaded FlashKDA decode {variant} {target} module")
    return module


def get_flash_kda_decode_module(
    variant: FlashKDADecodeVariant, target: FlashKDADecodeTarget
):
    """Return the loaded module used by the recurrent-KDA dispatcher."""

    return load_flash_kda_decode_module(variant, target)


__all__ = [
    "FLASH_KDA_DECODE_DIRECT_VARIANTS",
    "FLASH_KDA_DECODE_VARIANT_METADATA",
    "FLASH_KDA_DECODE_VARIANTS",
    "FlashKDADecodeTarget",
    "FlashKDADecodeVariant",
    "FlashKDADecodeVariantMetadata",
    "gen_flash_kda_decode_module",
    "get_flash_kda_decode_module",
    "get_flash_kda_decode_uri",
    "load_flash_kda_decode_module",
]
