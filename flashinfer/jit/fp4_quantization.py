"""
Copyright (c) 2025 by FlashInfer team.

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

from typing import List

from .core import JitSpec, gen_jit_spec
from . import env as jit_env
from .core import (
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm90a_nvcc_flags,
    sm110a_nvcc_flags,
    sm120a_nvcc_flags,
    sm121a_nvcc_flags,
)
from .cpp_ext import is_cuda_version_at_least


def gen_fp4_quantization_module(nvcc_flags: List[str], device_arch: str) -> JitSpec:
    return gen_jit_spec(
        f"fp4_quantization_{device_arch}",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/thop/fp4Quantize.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/tensorrt_llm/thop/fp4Op.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/kernels/quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        ],
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4" if is_cuda_version_at_least("12.8") else "",
        ],
        extra_cflags=[
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4" if is_cuda_version_at_least("12.8") else "",
        ],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
        ],
    )


def gen_fp4_quantization_sm100_module() -> JitSpec:
    return gen_fp4_quantization_module(sm100a_nvcc_flags, "100")


def gen_fp4_quantization_sm103_module() -> JitSpec:
    return gen_fp4_quantization_module(sm103a_nvcc_flags, "103")


def gen_fp4_quantization_sm90_module() -> JitSpec:
    return gen_fp4_quantization_module(sm90a_nvcc_flags, "90")


def gen_fp4_quantization_sm110_module() -> JitSpec:
    return gen_fp4_quantization_module(sm110a_nvcc_flags, "110")


def gen_fp4_quantization_sm120_module() -> JitSpec:
    return gen_fp4_quantization_module(sm120a_nvcc_flags, "120")


def gen_fp4_quantization_sm121_module() -> JitSpec:
    return gen_fp4_quantization_module(sm121a_nvcc_flags, "121")
