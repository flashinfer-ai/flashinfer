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

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags


def _wan_hybrid_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "wan_hybrid"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "wan_hybrid"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("Wan hybrid CUDA sources are unavailable")


def _wan_hybrid_target(
    device: torch.device | str | int,
) -> tuple[str, list[str]]:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 0):
        return "sm100", sm100a_nvcc_flags
    if capability == (10, 3):
        return "sm103", sm103a_nvcc_flags
    raise ValueError(
        "Wan hybrid kernels require compute capability 10.0 or 10.3, "
        f"got {capability[0]}.{capability[1]}"
    )


@functools.cache
def gen_wan_hybrid_quantization_module(target: str) -> JitSpec:
    if target == "sm100":
        nvcc_flags = sm100a_nvcc_flags
        target_minor = "0"
    elif target == "sm103":
        nvcc_flags = sm103a_nvcc_flags
        target_minor = "3"
    else:
        raise ValueError(f"Unsupported Wan hybrid quantization target: {target!r}")
    csrc_dir = _wan_hybrid_csrc_dir()
    return gen_jit_spec(
        f"wan_hybrid_quantization_{target}",
        [csrc_dir / "wan_hybrid_quantization_binding.cu"],
        extra_cuda_cflags=[
            *nvcc_flags,
            "-DFLASHINFER_ENABLE_BF16",
            f"-DFLASHINFER_WAN_HYBRID_TARGET_MINOR={target_minor}",
        ],
        extra_include_paths=[csrc_dir, csrc_dir.parent],
        use_fast_math=False,
    )


@functools.cache
def gen_wan_hybrid_attention_module(target: str) -> JitSpec:
    if target == "sm100":
        nvcc_flags = sm100a_nvcc_flags
        target_minor = "0"
    elif target == "sm103":
        nvcc_flags = sm103a_nvcc_flags
        target_minor = "3"
    else:
        raise ValueError(f"Unsupported Wan hybrid attention target: {target!r}")
    csrc_dir = _wan_hybrid_csrc_dir()
    return gen_jit_spec(
        f"wan_hybrid_attention_{target}",
        [csrc_dir / "wan_hybrid_attention_binding.cu"],
        extra_cuda_cflags=[
            *nvcc_flags,
            "-DFLASHINFER_ENABLE_BF16",
            f"-DFLASHINFER_WAN_HYBRID_TARGET_MINOR={target_minor}",
            "--ptxas-options=--opt-level=1",
        ],
        extra_include_paths=[csrc_dir, csrc_dir.parent],
        use_fast_math=False,
    )


@functools.cache
def gen_wan_hybrid_dispatch_module(target: str) -> JitSpec:
    if target == "sm100":
        nvcc_flags = sm100a_nvcc_flags
        target_minor = "0"
    elif target == "sm103":
        nvcc_flags = sm103a_nvcc_flags
        target_minor = "3"
    else:
        raise ValueError(f"Unsupported Wan hybrid dispatch target: {target!r}")
    csrc_dir = _wan_hybrid_csrc_dir()
    return gen_jit_spec(
        f"wan_hybrid_dispatch_{target}",
        [csrc_dir / "wan_hybrid_dispatch_binding.cu"],
        extra_cuda_cflags=[
            *nvcc_flags,
            "-DFLASHINFER_ENABLE_BF16",
            f"-DFLASHINFER_WAN_HYBRID_TARGET_MINOR={target_minor}",
            "--ptxas-options=--opt-level=1",
        ],
        extra_include_paths=[csrc_dir, csrc_dir.parent],
        use_fast_math=False,
    )
