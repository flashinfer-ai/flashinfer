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

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal, NamedTuple

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags
from .utils import write_if_different

BlackwellBGMVMoEDType = Literal["bfloat16", "float16"]
BlackwellBGMVMoESchedule = Literal[
    "pair_owned_t128",
    "token_owned_t64",
    "token_owned",
    "token_owned_dual_col",
]

BLACKWELL_BGMV_MOE_HIDDEN_SIZES = (2688, 3072)
BLACKWELL_BGMV_MOE_DTYPES: tuple[BlackwellBGMVMoEDType, ...] = (
    "bfloat16",
    "float16",
)
BLACKWELL_BGMV_MOE_SCHEDULE_IDS: dict[BlackwellBGMVMoESchedule, int] = {
    "pair_owned_t128": 0,
    "token_owned_t64": 1,
    "token_owned": 2,
    "token_owned_dual_col": 3,
}


class BlackwellBGMVMoEMetadata(NamedTuple):
    body: str
    shrink_decode_symbol: str
    shrink_prefill_symbol: str
    pair_owned_symbol: str
    token_t64_symbol: str
    token_symbol: str
    token_dual_col_symbol: str


def _dtype_tag(dtype: BlackwellBGMVMoEDType) -> str:
    if dtype == "bfloat16":
        return "bf16"
    if dtype == "float16":
        return "f16"
    raise ValueError(f"unsupported Blackwell BGMV MoE dtype: {dtype}")


def _metadata(
    hidden_size: int, dtype: BlackwellBGMVMoEDType
) -> BlackwellBGMVMoEMetadata:
    if hidden_size not in BLACKWELL_BGMV_MOE_HIDDEN_SIZES:
        raise ValueError(
            f"Blackwell BGMV MoE hidden_size must be 2688 or 3072, got {hidden_size}"
        )
    tag = _dtype_tag(dtype)
    return BlackwellBGMVMoEMetadata(
        body=f"blackwell_bgmv_moe_{tag}_h{hidden_size}_sm100a.cu",
        shrink_decode_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_{tag}_h{hidden_size}_r32_p4_s3"
        ),
        shrink_prefill_symbol=(
            f"kernel_flashinfer_bgmv_moe_shrink_{tag}_h{hidden_size}_r32_p1_s2"
        ),
        pair_owned_symbol=(
            "kernel_flashinfer_bgmv_moe_expand_pair_owned_"
            f"{tag}_h{hidden_size}_r32_t128"
        ),
        token_t64_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_token_t64_{tag}_h{hidden_size}_r32"
        ),
        token_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_token_{tag}_h{hidden_size}_r32"
        ),
        token_dual_col_symbol=(
            f"kernel_flashinfer_bgmv_moe_expand_token_dual_col_{tag}_h{hidden_size}_r32"
        ),
    )


def select_blackwell_bgmv_moe_schedule(
    hidden_size: int,
    num_tokens: int,
) -> BlackwellBGMVMoESchedule:
    """Return the measured selector for the supported rank-32 portfolio."""

    if hidden_size not in BLACKWELL_BGMV_MOE_HIDDEN_SIZES:
        raise ValueError(
            f"Blackwell BGMV MoE hidden_size must be 2688 or 3072, got {hidden_size}"
        )
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    if (hidden_size == 3072 and num_tokens in (1, 4, 8)) or (
        hidden_size == 2688 and num_tokens in (1, 8)
    ):
        return "token_owned_t64"
    if hidden_size == 2688 and num_tokens == 4:
        return "pair_owned_t128"
    if hidden_size == 3072 and num_tokens in (512, 1024):
        return "token_owned_dual_col"
    if hidden_size == 2688 and num_tokens == 1024:
        return "token_owned_dual_col"
    return "token_owned"


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "blackwell_bgmv_moe" / "sm100a"
    if installed.is_dir():
        return installed
    checkout = (
        Path(__file__).resolve().parents[2] / "csrc" / "blackwell_bgmv_moe" / "sm100a"
    )
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "generated Blackwell BGMV MoE sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
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


def get_blackwell_bgmv_moe_uri(
    hidden_size: int,
    dtype: BlackwellBGMVMoEDType,
) -> str:
    tag = _dtype_tag(dtype)
    if hidden_size not in BLACKWELL_BGMV_MOE_HIDDEN_SIZES:
        raise ValueError(
            f"Blackwell BGMV MoE hidden_size must be 2688 or 3072, got {hidden_size}"
        )
    return f"blackwell_bgmv_moe_{tag}_h{hidden_size}_sm100a"


def _binding_source(metadata: BlackwellBGMVMoEMetadata, hidden_size: int) -> str:
    input_dtype = "dl_bfloat16" if "_bf16_" in metadata.body else "dl_float16"
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#define BLACKWELL_BGMV_MOE_BODY_FILE \"{metadata.body}\"
#define BLACKWELL_BGMV_MOE_HIDDEN {hidden_size}
#define BLACKWELL_BGMV_MOE_INPUT_DTYPE {input_dtype}
#define BLACKWELL_BGMV_MOE_SHRINK_DECODE {metadata.shrink_decode_symbol}
#define BLACKWELL_BGMV_MOE_SHRINK_PREFILL {metadata.shrink_prefill_symbol}
#define BLACKWELL_BGMV_MOE_EXPAND_PAIR {metadata.pair_owned_symbol}
#define BLACKWELL_BGMV_MOE_EXPAND_TOKEN_T64 {metadata.token_t64_symbol}
#define BLACKWELL_BGMV_MOE_EXPAND_TOKEN {metadata.token_symbol}
#define BLACKWELL_BGMV_MOE_EXPAND_TOKEN_DUAL {metadata.token_dual_col_symbol}

#include \"blackwell_bgmv_moe_binding.cuh\"
"""


@functools.cache
def gen_blackwell_bgmv_moe_module(
    hidden_size: int,
    dtype: BlackwellBGMVMoEDType,
) -> JitSpec:
    metadata = _metadata(hidden_size, dtype)
    csrc_dir = _get_csrc_dir()
    include_dir = _get_include_dir()
    body = csrc_dir / metadata.body
    binding_header = csrc_dir / "blackwell_bgmv_moe_binding.cuh"
    if not body.is_file():
        raise FileNotFoundError(f"generated Blackwell BGMV MoE body not found: {body}")
    if not binding_header.is_file():
        raise FileNotFoundError(
            f"Blackwell BGMV MoE binding header not found: {binding_header}"
        )

    uri = get_blackwell_bgmv_moe_uri(hidden_size, dtype)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "blackwell_bgmv_moe_binding.cu"
    write_if_different(binding, _binding_source(metadata, hidden_size))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[*sm100a_nvcc_flags, "-use_fast_math"],
        extra_include_paths=[csrc_dir, include_dir],
    )
    logger.info("Generated Blackwell BGMV MoE JIT spec: %s", spec.name)
    return spec


@functools.cache
def load_blackwell_bgmv_moe_module(
    hidden_size: int,
    dtype: BlackwellBGMVMoEDType,
):
    module = gen_blackwell_bgmv_moe_module(hidden_size, dtype).build_and_load()
    logger.info(
        "Loaded Blackwell BGMV MoE module for hidden_size=%d, dtype=%s",
        hidden_size,
        dtype,
    )
    return module


def get_blackwell_bgmv_moe_module(
    hidden_size: int,
    dtype: BlackwellBGMVMoEDType,
):
    return load_blackwell_bgmv_moe_module(hidden_size, dtype)


__all__ = [
    "BLACKWELL_BGMV_MOE_DTYPES",
    "BLACKWELL_BGMV_MOE_HIDDEN_SIZES",
    "BLACKWELL_BGMV_MOE_SCHEDULE_IDS",
    "BlackwellBGMVMoEDType",
    "BlackwellBGMVMoEMetadata",
    "BlackwellBGMVMoESchedule",
    "gen_blackwell_bgmv_moe_module",
    "get_blackwell_bgmv_moe_module",
    "get_blackwell_bgmv_moe_uri",
    "load_blackwell_bgmv_moe_module",
    "select_blackwell_bgmv_moe_schedule",
]
