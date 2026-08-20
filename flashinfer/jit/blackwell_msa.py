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

BlackwellMSAVariant = Literal[
    "decode_m16_bf16_flat",
    "decode_m16_bf16_paged",
    "decode_m16_bf16_query_fp8_kv_flat",
    "decode_m16_bf16_query_fp8_kv_paged",
    "decode_m16_fp16_flat",
    "decode_m16_fp16_paged",
    "decode_q1_bf16_query_fp8_kv_exact_flat",
    "decode_q1_bf16_query_fp8_kv_exact_paged",
    "decode_q1_bf16_query_fp8_kv_xform2_flat",
    "decode_q1_bf16_query_fp8_kv_xform2_paged",
    "decode_uniform_fp8_qkv_paged",
    "long_prefill_flat_bf16_gqa16_sm100",
    "long_prefill_flat_bf16_gqa16_sm103",
    "long_prefill_paged_bf16_gqa16_direct_group_sm100",
    "long_prefill_paged_bf16_gqa16_sm100",
    "long_prefill_paged_bf16_gqa16_sm103",
    "long_prefill_paged_bf16_gqa8_sm100",
    "long_prefill_paged_bf16_gqa8_sm103",
    "long_prefill_reduce_flat_bf16_gqa16",
    "long_prefill_reduce_paged_bf16_gqa16",
    "long_prefill_reduce_paged_bf16_gqa8",
    "prefill_m64_bf16_gqa16_flat",
    "prefill_union_bf16_flat",
    "prefill_union_bf16_gqa16_flat",
    "prefill_union_bf16_gqa16_paged_causal_large",
    "prefill_union_bf16_gqa16_paged_causal_mask64",
    "prefill_union_bf16_gqa16_paged_noncausal",
    "prefill_union_bf16_gqa8_flat",
    "prefill_union_bf16_gqa8_paged",
    "prefill_union_bf16_paged",
    "prefill_union_bf16_query_fp8_kv_flat",
    "prefill_union_bf16_query_fp8_kv_paged",
    "prefill_union_fp16_flat",
    "prefill_union_fp16_paged",
    "topk",
]
BlackwellMSATarget = Literal["sm100a", "sm103a"]

_COMMON_VARIANTS: tuple[BlackwellMSAVariant, ...] = (
    "decode_m16_bf16_flat",
    "decode_m16_bf16_paged",
    "decode_m16_bf16_query_fp8_kv_flat",
    "decode_m16_bf16_query_fp8_kv_paged",
    "decode_m16_fp16_flat",
    "decode_m16_fp16_paged",
    "decode_q1_bf16_query_fp8_kv_exact_flat",
    "decode_q1_bf16_query_fp8_kv_exact_paged",
    "decode_q1_bf16_query_fp8_kv_xform2_flat",
    "decode_q1_bf16_query_fp8_kv_xform2_paged",
    "decode_uniform_fp8_qkv_paged",
    "long_prefill_reduce_flat_bf16_gqa16",
    "long_prefill_reduce_paged_bf16_gqa16",
    "long_prefill_reduce_paged_bf16_gqa8",
    "prefill_m64_bf16_gqa16_flat",
    "prefill_union_bf16_flat",
    "prefill_union_bf16_gqa16_flat",
    "prefill_union_bf16_gqa16_paged_causal_large",
    "prefill_union_bf16_gqa16_paged_causal_mask64",
    "prefill_union_bf16_gqa16_paged_noncausal",
    "prefill_union_bf16_gqa8_flat",
    "prefill_union_bf16_gqa8_paged",
    "prefill_union_bf16_paged",
    "prefill_union_bf16_query_fp8_kv_flat",
    "prefill_union_bf16_query_fp8_kv_paged",
    "prefill_union_fp16_flat",
    "prefill_union_fp16_paged",
    "topk",
)

BLACKWELL_MSA_VARIANTS_BY_TARGET: dict[
    BlackwellMSATarget, tuple[BlackwellMSAVariant, ...]
] = {
    "sm100a": tuple(
        sorted(
            (
                *_COMMON_VARIANTS,
                "long_prefill_flat_bf16_gqa16_sm100",
                "long_prefill_paged_bf16_gqa16_direct_group_sm100",
                "long_prefill_paged_bf16_gqa16_sm100",
                "long_prefill_paged_bf16_gqa8_sm100",
            )
        )
    ),
    "sm103a": tuple(
        sorted(
            (
                *_COMMON_VARIANTS,
                "long_prefill_flat_bf16_gqa16_sm103",
                "long_prefill_paged_bf16_gqa16_sm103",
                "long_prefill_paged_bf16_gqa8_sm103",
            )
        )
    ),
}
BLACKWELL_MSA_VARIANTS: tuple[BlackwellMSAVariant, ...] = tuple(
    sorted(
        set(BLACKWELL_MSA_VARIANTS_BY_TARGET["sm100a"])
        | set(BLACKWELL_MSA_VARIANTS_BY_TARGET["sm103a"])
    )
)

_BLACKWELL_MSA_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_BLACKWELL_MSA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_BLACKWELL_MSA_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_BLACKWELL_MSA_TARGET_MINOR=3",
}


def _get_blackwell_msa_csrc_dir() -> Path:
    """Locate Blackwell MSA CUDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "blackwell_msa"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "blackwell_msa"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "Blackwell MSA CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_blackwell_msa_include_dir() -> Path:
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


def get_blackwell_msa_uri(variant: BlackwellMSAVariant, target: BlackwellMSATarget) -> str:
    """Return the target-specific JIT/AOT key for one Blackwell MSA variant."""

    if target not in _BLACKWELL_MSA_NVCC_FLAGS:
        raise ValueError(f"unsupported Blackwell MSA target: {target}")
    if variant not in BLACKWELL_MSA_VARIANTS_BY_TARGET[target]:
        raise ValueError(f"unsupported Blackwell MSA variant/target: {variant}/{target}")
    return f"blackwell_msa_{variant}_{target}"


@functools.cache
def gen_blackwell_msa_module(variant: BlackwellMSAVariant, target: BlackwellMSATarget) -> JitSpec:
    """Generate one exact-SM100a or exact-SM103a Blackwell MSA JIT module."""

    csrc_dir = _get_blackwell_msa_csrc_dir()
    include_dir = _get_blackwell_msa_include_dir()
    uri = get_blackwell_msa_uri(variant, target)
    target_dir = csrc_dir / target
    body = target_dir / f"blackwell_msa_{variant}.cu"
    binding = target_dir / f"blackwell_msa_{variant}_binding.cu"
    if not body.exists():
        raise FileNotFoundError(f"Blackwell MSA CUDA source not found: {body}")
    if not binding.exists():
        raise FileNotFoundError(f"Blackwell MSA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_BLACKWELL_MSA_NVCC_FLAGS[target],
            _BLACKWELL_MSA_TARGET_DEFINE[target],
        ],
        extra_include_paths=[
            target_dir,
            csrc_dir.parent,
            include_dir,
        ],
    )
    logger.info(f"Generated Blackwell MSA {variant} {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def load_blackwell_msa_module(variant: BlackwellMSAVariant, target: BlackwellMSATarget):
    """Build or load one physical, target-specific Blackwell MSA module."""

    module = gen_blackwell_msa_module(variant, target).build_and_load()
    logger.info(f"Loaded Blackwell MSA {variant} {target} module")
    return module


def get_blackwell_msa_module(variant: BlackwellMSAVariant, target: BlackwellMSATarget):
    """Return the loaded module used by the MSA backend dispatcher."""

    return load_blackwell_msa_module(variant, target)


__all__ = [
    "BLACKWELL_MSA_VARIANTS",
    "BLACKWELL_MSA_VARIANTS_BY_TARGET",
    "BlackwellMSATarget",
    "BlackwellMSAVariant",
    "gen_blackwell_msa_module",
    "get_blackwell_msa_module",
    "get_blackwell_msa_uri",
    "load_blackwell_msa_module",
]
