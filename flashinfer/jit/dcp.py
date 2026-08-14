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
from typing import Literal

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
)

DcpSpecVariant = Literal["v1", "v4"]
DcpSpecTarget = Literal["sm100a", "sm100f"]

_DCP_SPEC_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
}
_SUPPORTED_Q_LENS = (1, 2, 4, 5, 6, 8)
_FP8_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_SUPPORTED_CP_WORLDS = (1, 2, 4, 8)


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "dcp"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "dcp"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "DCP speculative FMHA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


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
        f"_cp{cp_world}_{route_name}{route_param}"
    )


def _validate_fp8_specialization(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
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


def get_dcp_spec_fp8_uri(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
) -> str:
    _validate_fp8_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
    )
    return (
        f"cake_fmha_dcp_spec_bf16_fp8_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}_cp{cp_world}"
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
    csrc_dir = _get_csrc_dir()
    route_name = "retain" if variant == "v1" else "split"
    body = csrc_dir / f"cake_fmha_dcp_spec_bf16_{variant}_{route_name}{route_param}.cu"
    binding = csrc_dir / f"cake_fmha_dcp_spec_bf16_{variant}_binding.cu"
    for source in (body, binding):
        if not source.exists():
            raise FileNotFoundError(f"DCP speculative FMHA source not found: {source}")

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
) -> JitSpec:
    """Generate one BF16-Q/FP8-KV, HND-page64 Cake FMHA module."""

    uri = get_dcp_spec_fp8_uri(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
    )
    csrc_dir = _get_csrc_dir()
    body = csrc_dir / "cake_fmha_dcp_spec_bf16_fp8.cu"
    binding = csrc_dir / "cake_fmha_dcp_spec_bf16_fp8_binding.cu"
    for source in (body, binding):
        if not source.exists():
            raise FileNotFoundError(f"DCP speculative FMHA source not found: {source}")

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
):
    module = gen_dcp_spec_fp8_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
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
