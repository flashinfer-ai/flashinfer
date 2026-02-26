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

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    current_compilation_context,
)


def gen_moe_utils_module() -> JitSpec:
    """
    Generate a JitSpec for the MoE utilities module.

    This module contains:
    - moePermute: Permute input activations for MoE routing
    - moeUnpermute: Unpermute and scale outputs after expert computation
    - moeOutputMemset: Zero-initialize output buffers for scattered writes
    - moeActivation: Apply activation functions with optional FP4 quantization
    - moeSort: Sort tokens by expert assignment (DeepSeekV3 routing)
    """
    nvcc_flags = [
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
    ]

    nvcc_flags += current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]
    )

    return gen_jit_spec(
        "moe_utils",
        [
            jit_env.FLASHINFER_CSRC_DIR / "moe_utils_binding.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cuteDslKernels/moeUtils.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
            # Routing kernels for moe_sort
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_deepseek.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR,
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "cutlass_extensions"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels",
            # Include paths for routing kernels
            jit_env.FLASHINFER_INCLUDE_DIR,
            # Include path for trtllm/gen/MmaDecl.h (used by DtypeDecl.h)
            jit_env.FLASHINFER_INCLUDE_DIR
            / "flashinfer"
            / "trtllm"
            / "batched_gemm"
            / "trtllmGen_bmm_export",
        ],
    )
