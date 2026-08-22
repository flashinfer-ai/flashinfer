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

import functools
from typing import List

from .core import JitSpec, gen_jit_spec
from . import env as jit_env
from .core import (
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
    sm90a_nvcc_flags,
    sm110a_nvcc_flags,
    sm120a_nvcc_flags,
    sm120f_nvcc_flags,
    sm121a_nvcc_flags,
)
from .cpp_ext import (
    has_prebuilt_aot_module,
    is_cuda_version_at_least,
    version_gated_nvcc_flag,
)


@functools.cache
def has_fp4_support(device_arch: str = "100") -> bool:
    """Whether the fp4_quantization module for ``device_arch`` contains the
    kernels gated by ``-DENABLE_FP4``.

    They are compiled only when ``-DENABLE_FP4`` survives flag generation,
    i.e. when a CUDA 12.8+ toolkit is used for JIT compilation or when a
    prebuilt AOT module (built with a 12.8+ toolkit, e.g. from
    flashinfer-jit-cache) is available. Public capability query requested in
    issue #3951 so frameworks can gate fp4 paths before committing to a JIT
    build. ``device_arch`` is the compute capability as a string, e.g. "100"
    for SM100, matching ``get_fp4_quantization_module``.
    """
    if has_prebuilt_aot_module(f"fp4_quantization_{device_arch}"):
        return True
    return is_cuda_version_at_least("12.8")


def gen_fp4_quantization_module(nvcc_flags: List[str], device_arch: str) -> JitSpec:
    enable_fp4_flag = version_gated_nvcc_flag(
        "-DENABLE_FP4", "12.8", f"fp4_quantization_{device_arch}"
    )
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
            enable_fp4_flag,
        ],
        extra_cflags=[
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            enable_fp4_flag,
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


def gen_fp4_quantization_sm107_module() -> JitSpec:
    from flashinfer.compilation_context import cutlass_supports_sm107

    nvcc_flags = sm107a_nvcc_flags if cutlass_supports_sm107() else sm100f_nvcc_flags
    return gen_fp4_quantization_module(nvcc_flags, "107")


def gen_fp4_quantization_sm90_module() -> JitSpec:
    return gen_fp4_quantization_module(sm90a_nvcc_flags, "90")


def gen_fp4_quantization_sm110_module() -> JitSpec:
    return gen_fp4_quantization_module(sm110a_nvcc_flags, "110")


def gen_fp4_quantization_sm120_module() -> JitSpec:
    return gen_fp4_quantization_module(sm120a_nvcc_flags, "120")


def gen_fp4_quantization_sm120f_module() -> JitSpec:
    return gen_fp4_quantization_module(sm120f_nvcc_flags, "120f")


def gen_fp4_quantization_sm121_module() -> JitSpec:
    return gen_fp4_quantization_module(sm121a_nvcc_flags, "121")
