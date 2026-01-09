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

from .. import env as jit_env
from ..core import JitSpec, common_nvcc_flags, gen_jit_spec


def gen_selective_state_update_module() -> JitSpec:
    nvcc_flags = [
        "-DENABLE_BF16",
    ]

    return gen_jit_spec(
        "mamba_selective_state_update",
        [
            jit_env.FLASHINFER_CSRC_DIR / "selective_state_update.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mamba_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )


def gen_selective_state_update_sm90_module() -> JitSpec:
    # Use generic SM90 flags to support SM90, SM100 and future architectures
    # code=compute_90 embeds PTX for forward compatibility
    nvcc_flags = [
        "-gencode=arch=compute_90,code=[sm_90,compute_90]",
        "-DENABLE_BF16",
        "-DFLASHINFER_MAMBA_ENABLE_SM90",
    ] + common_nvcc_flags

    return gen_jit_spec(
        "mamba_selective_state_update_sm90",
        [
            jit_env.FLASHINFER_CSRC_DIR / "selective_state_update.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mamba_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )
