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
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from .cake_fmha import (
    CAKE_FMHA_FLASHINFER_BINDINGS_SHA256,
    CAKE_FMHA_MANIFEST_SHA256,
    get_cake_fmha_csrc_dir,
    get_cake_fmha_manifest,
)
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

DcpSpecVariant = Literal["v1", "v4"]
DcpSpecTarget = Literal["sm100a", "sm103a"]

_DCP_SPEC_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_TARGET_MANIFEST_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_DCP_JIT_BINDINGS = {
    "dcp_spec_bf16_v1": "jit/cake_fmha_dcp_spec_bf16_v1_jit_binding.cu",
    "dcp_spec_bf16_v4": "jit/cake_fmha_dcp_spec_bf16_v4_jit_binding.cu",
    "dcp_spec_bf16_fp8": "jit/cake_fmha_dcp_spec_bf16_fp8_jit_binding.cu",
}
_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_FP8_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_SUPPORTED_CP_WORLDS = (1, 2, 4, 8)


def _get_dcp_family(name: str) -> Mapping[str, object]:
    addon = get_cake_fmha_manifest()["add_ons"]["cake_fmha_dcp_spec"]
    if addon.get("installed") is not True:
        raise RuntimeError("the authenticated Cake FMHA DCP add-on is not installed")
    families = addon["manifest"]["families"]
    try:
        return families[name]
    except KeyError as exc:
        raise RuntimeError(f"Cake FMHA DCP family is missing: {name}") from exc


def _get_dcp_sources(
    family_name: str,
    target: DcpSpecTarget,
    selector: Mapping[str, int],
) -> tuple[Path, Path]:
    family = _get_dcp_family(family_name)
    matches = [
        entry
        for entry in family["source_family"]
        if entry.get("selector") == dict(selector)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Cake FMHA DCP selector is not unique: {family_name} {dict(selector)!r}"
        )
    csrc_dir = get_cake_fmha_csrc_dir()
    body = csrc_dir / matches[0]["sources"][_TARGET_MANIFEST_ARCH[target]]
    binding = csrc_dir / _DCP_JIT_BINDINGS[family_name]
    for source in (body, binding):
        if not source.is_file():
            raise FileNotFoundError(f"Cake FMHA DCP source not found: {source}")
    return body, binding


def _validate_specialization(
    variant: DcpSpecVariant,
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    route_param: int,
) -> None:
    if variant not in ("v1", "v4"):
        raise ValueError(f"unsupported DCP speculative FMHA variant: {variant}")
    if target not in _DCP_SPEC_NVCC_FLAGS:
        raise ValueError(f"unsupported DCP speculative FMHA target: {target}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if q_len not in _SUPPORTED_Q_LENS:
        raise ValueError(f"q_len must be one of {_SUPPORTED_Q_LENS}, got {q_len}")
    if cp_world not in _SUPPORTED_CP_WORLDS:
        raise ValueError(
            f"cp_world must be one of {_SUPPORTED_CP_WORLDS}, got {cp_world}"
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads for GQA: "
            f"got {num_q_heads} and {num_kv_heads}"
        )
    group_ratio = num_q_heads // num_kv_heads
    if not 1 <= group_ratio <= 8:
        raise ValueError(f"head group ratio must be in [1, 8], got {group_ratio}")
    if variant == "v1" and route_param not in (0, 1):
        raise ValueError(f"v1 retain_kv_l2 must be 0 or 1, got {route_param}")
    if variant == "v4" and not 2 <= route_param <= 16:
        raise ValueError(f"v4 num_split must be in [2, 16], got {route_param}")


def get_dcp_spec_uri(
    variant: DcpSpecVariant,
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    route_param: int,
) -> str:
    _validate_specialization(
        variant,
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        route_param,
    )
    route_name = "retain" if variant == "v1" else "split"
    return (
        f"cake_fmha_dcp_spec_bf16_{variant}_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_cp{cp_world}_{route_name}{route_param}_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


def _validate_fp8_specialization(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
    retain_kv_l2: int,
) -> None:
    if target not in _DCP_SPEC_NVCC_FLAGS:
        raise ValueError(f"unsupported DCP speculative FMHA target: {target}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if q_len not in _FP8_SUPPORTED_Q_LENS:
        raise ValueError(f"q_len must be one of {_FP8_SUPPORTED_Q_LENS}, got {q_len}")
    if cp_world not in _SUPPORTED_CP_WORLDS:
        raise ValueError(
            f"cp_world must be one of {_SUPPORTED_CP_WORLDS}, got {cp_world}"
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads for GQA: "
            f"got {num_q_heads} and {num_kv_heads}"
        )
    group_ratio = num_q_heads // num_kv_heads
    if not 1 <= group_ratio <= 8:
        raise ValueError(f"head group ratio must be in [1, 8], got {group_ratio}")
    if not 1 <= num_split <= 4:
        raise ValueError(f"FP8 num_split must be in [1, 4], got {num_split}")
    if retain_kv_l2 not in (0, 1):
        raise ValueError(f"FP8 retain_kv_l2 must be 0 or 1, got {retain_kv_l2}")


def get_dcp_spec_fp8_uri(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
    retain_kv_l2: int,
) -> str:
    _validate_fp8_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
        retain_kv_l2,
    )
    return (
        f"cake_fmha_dcp_spec_bf16_fp8_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}_cp{cp_world}"
        f"_split{num_split}_retain{retain_kv_l2}_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_dcp_spec_module(
    variant: DcpSpecVariant,
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    route_param: int,
) -> JitSpec:
    """Generate one source-specialized, one-launch DCP speculative FMHA module."""

    uri = get_dcp_spec_uri(
        variant,
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        route_param,
    )
    selector = (
        {"retain_kv_l2": route_param} if variant == "v1" else {"num_split": route_param}
    )
    body, binding = _get_dcp_sources(f"dcp_spec_bf16_{variant}", target, selector)
    csrc_dir = get_cake_fmha_csrc_dir()

    spec = gen_jit_spec(
        name=uri,
        sources=[body, binding],
        extra_cuda_cflags=[
            *_DCP_SPEC_NVCC_FLAGS[target],
            f"-DBATCH_SIZE={batch_size}",
            f"-DQ_LEN={q_len}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCP_WORLD={cp_world}",
        ],
        extra_include_paths=[csrc_dir],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated DCP speculative FMHA JIT spec: {spec.name}")
    return spec


@functools.cache
def gen_dcp_spec_fp8_module(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
    retain_kv_l2: int,
) -> JitSpec:
    """Generate one BF16-Q/FP8-KV, HND-page64 Cake FMHA module."""

    uri = get_dcp_spec_fp8_uri(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
        retain_kv_l2,
    )
    body, binding = _get_dcp_sources(
        "dcp_spec_bf16_fp8",
        target,
        {"num_split": num_split, "retain_kv_l2": retain_kv_l2},
    )
    csrc_dir = get_cake_fmha_csrc_dir()

    spec = gen_jit_spec(
        name=uri,
        sources=[body, binding],
        extra_cuda_cflags=[
            *_DCP_SPEC_NVCC_FLAGS[target],
            f"-DBATCH_SIZE={batch_size}",
            f"-DQ_LEN={q_len}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCP_WORLD={cp_world}",
        ],
        extra_include_paths=[csrc_dir],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated FP8 DCP speculative FMHA JIT spec: {spec.name}")
    return spec


@functools.cache
def load_dcp_spec_module(
    variant: DcpSpecVariant,
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    route_param: int,
):
    module = gen_dcp_spec_module(
        variant,
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        route_param,
    ).build_and_load()
    logger.info(f"Loaded DCP speculative FMHA module: {module}")
    return module


@functools.cache
def load_dcp_spec_fp8_module(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
    retain_kv_l2: int,
):
    module = gen_dcp_spec_fp8_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
        retain_kv_l2,
    ).build_and_load()
    logger.info(f"Loaded FP8 DCP speculative FMHA module: {module}")
    return module


__all__ = [
    "DcpSpecTarget",
    "DcpSpecVariant",
    "gen_dcp_spec_fp8_module",
    "gen_dcp_spec_module",
    "get_dcp_spec_fp8_uri",
    "get_dcp_spec_uri",
    "load_dcp_spec_fp8_module",
    "load_dcp_spec_module",
]
