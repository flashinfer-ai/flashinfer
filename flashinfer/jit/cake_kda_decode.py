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

CakeKDADecodeVariant = Literal[
    "d128_t1_unbounded_softplus_direct_split4",
    "d128_t1_unbounded_softplus_direct_split16",
    "d128_t1_unbounded_softplus_direct_split8",
    "d128_t1_unbounded_softplus_direct_split8_warp2",
]
CakeKDADecodeTarget = Literal["sm100a", "sm100f", "sm103a"]

_CAKE_KDA_DECODE_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}

# The binding uses one numeric target kind to enforce the same execution
# boundary for JIT and AOT modules. 100 is the CC 10.0/10.3 family target;
# 1000 and 1003 are exact CC targets used by the CUDA-12.8 compatibility path
# and the three GB300 direct-T1 performance specializations, respectively.
_CAKE_KDA_DECODE_TARGET_KIND = {
    "sm100f": 100,
    "sm100a": 1000,
    "sm103a": 1003,
}
CAKE_KDA_DECODE_VARIANTS: tuple[CakeKDADecodeVariant, ...] = (
    "d128_t1_unbounded_softplus_direct_split4",
    "d128_t1_unbounded_softplus_direct_split16",
    "d128_t1_unbounded_softplus_direct_split8",
    "d128_t1_unbounded_softplus_direct_split8_warp2",
)

CAKE_KDA_DECODE_DIRECT_VARIANTS: tuple[CakeKDADecodeVariant, ...] = (
    "d128_t1_unbounded_softplus_direct_split4",
    "d128_t1_unbounded_softplus_direct_split16",
    "d128_t1_unbounded_softplus_direct_split8",
    "d128_t1_unbounded_softplus_direct_split8_warp2",
)


class CakeKDADecodeVariantMetadata(NamedTuple):
    head_dim: int
    tokens: int
    gate_kind: int
    value_split: int
    launch_threads: int
    warps_per_cta: int
    direct_impl: bool


def _variant_metadata(
    tokens: int,
    gate_kind: int,
    value_split: int,
    coefficient_gram: bool = False,
    direct_impl: bool = False,
    warps_per_cta: int = 1,
) -> CakeKDADecodeVariantMetadata:
    """Derive the exact launch geometry used by the frozen Cake schedule."""

    head_dim = 128
    value_rows = head_dim // value_split
    value_warps = value_rows // 16
    rows_per_group = 2 if coefficient_gram and value_split == 8 else 8
    state_warps = (value_rows // rows_per_group + 1) // 2
    launch_threads = (
        32 * warps_per_cta
        if direct_impl
        else max(tokens, value_warps, state_warps) * 32
    )
    return CakeKDADecodeVariantMetadata(
        head_dim,
        tokens,
        gate_kind,
        value_split,
        launch_threads,
        warps_per_cta,
        direct_impl,
    )


CAKE_KDA_DECODE_VARIANT_METADATA: dict[
    CakeKDADecodeVariant, CakeKDADecodeVariantMetadata
] = {
    "d128_t1_unbounded_softplus_direct_split4": _variant_metadata(
        1, 2, 4, direct_impl=True
    ),
    "d128_t1_unbounded_softplus_direct_split16": _variant_metadata(
        1, 2, 16, direct_impl=True
    ),
    "d128_t1_unbounded_softplus_direct_split8": _variant_metadata(
        1, 2, 8, direct_impl=True
    ),
    "d128_t1_unbounded_softplus_direct_split8_warp2": _variant_metadata(
        1, 2, 8, direct_impl=True, warps_per_cta=2
    ),
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
        "frozen CakeKDA decode sources were not found. Checked:\n"
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


def get_cake_kda_decode_uri(
    variant: CakeKDADecodeVariant, target: CakeKDADecodeTarget
) -> str:
    """Return the physical-target JIT/AOT key for one decode schedule."""

    if variant not in CAKE_KDA_DECODE_VARIANTS:
        raise ValueError(f"unsupported CakeKDA decode variant: {variant}")
    if target not in _CAKE_KDA_DECODE_NVCC_FLAGS:
        raise ValueError(f"unsupported CakeKDA decode target: {target}")
    if target == "sm103a" and variant not in CAKE_KDA_DECODE_DIRECT_VARIANTS:
        raise ValueError(
            "exact SM103a CakeKDA decode modules are only retained for "
            f"direct T=1 variants, got {variant}"
        )
    return f"cake_kda_decode_{variant}_{target}"


def _get_binding_cu(
    variant: CakeKDADecodeVariant,
    metadata: CakeKDADecodeVariantMetadata,
) -> str:
    """Render the generic binding translation unit for one frozen body."""

    body_file = f"cake_kda_decode_{variant}.cu"
    direct_impl_define = (
        "#define CAKE_KDA_DECODE_DIRECT_IMPL 1\n" if metadata.direct_impl else ""
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

#define CAKE_KDA_DECODE_BODY_FILE "{body_file}"
#define CAKE_KDA_DECODE_HEAD_DIM {metadata.head_dim}
#define CAKE_KDA_DECODE_TOKENS {metadata.tokens}
#define CAKE_KDA_DECODE_GATE_KIND {metadata.gate_kind}
#define CAKE_KDA_DECODE_VALUE_SPLIT {metadata.value_split}
#define CAKE_KDA_DECODE_LAUNCH_THREADS {metadata.launch_threads}
#define CAKE_KDA_DECODE_WARPS_PER_CTA {metadata.warps_per_cta}
{direct_impl_define}

#include "cake_kda_decode_binding.cuh"
"""


@functools.cache
def gen_cake_kda_decode_module(
    variant: CakeKDADecodeVariant, target: CakeKDADecodeTarget
) -> JitSpec:
    """Generate one family or exact-target frozen decode module.

    ``sm100f`` is the normal CUDA-12.9+ target for both CC 10.0 and CC 10.3.
    ``sm100a`` preserves CUDA-12.8 B200 support, while ``sm103a`` is limited to
    the direct T=1 bodies whose family-target code showed a measurable
    GB300 latency regression.
    """

    csrc_dir = _get_csrc_dir()
    body = csrc_dir / f"cake_kda_decode_{variant}.cu"
    if not body.exists():
        raise FileNotFoundError(f"frozen CakeKDA decode body source not found: {body}")
    binding_header = csrc_dir / "cake_kda_decode_binding.cuh"
    if not binding_header.exists():
        raise FileNotFoundError(
            f"generic CakeKDA decode binding header not found: {binding_header}"
        )

    metadata = CAKE_KDA_DECODE_VARIANT_METADATA[variant]
    uri = get_cake_kda_decode_uri(variant, target)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_kda_decode_binding.cu"
    write_if_different(binding, _get_binding_cu(variant, metadata))

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_CAKE_KDA_DECODE_NVCC_FLAGS[target],
            (
                "-DFLASHINFER_CAKE_KDA_DECODE_TARGET_KIND="
                f"{_CAKE_KDA_DECODE_TARGET_KIND[target]}"
            ),
            "--maxrregcount=128",
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_include_dir(),
        ],
    )
    logger.info(f"Generated CakeKDA decode {variant} {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def load_cake_kda_decode_module(
    variant: CakeKDADecodeVariant, target: CakeKDADecodeTarget
):
    """Build or load one physical-target decode module."""

    module = gen_cake_kda_decode_module(variant, target).build_and_load()
    logger.info(f"Loaded CakeKDA decode {variant} {target} module")
    return module


def get_cake_kda_decode_module(
    variant: CakeKDADecodeVariant, target: CakeKDADecodeTarget
):
    """Return the loaded module used by the recurrent-KDA dispatcher."""

    return load_cake_kda_decode_module(variant, target)


__all__ = [
    "CAKE_KDA_DECODE_DIRECT_VARIANTS",
    "CAKE_KDA_DECODE_VARIANT_METADATA",
    "CAKE_KDA_DECODE_VARIANTS",
    "CakeKDADecodeTarget",
    "CakeKDADecodeVariant",
    "CakeKDADecodeVariantMetadata",
    "gen_cake_kda_decode_module",
    "get_cake_kda_decode_module",
    "get_cake_kda_decode_uri",
    "load_cake_kda_decode_module",
]
