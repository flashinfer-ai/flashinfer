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

from ...compilation_context import CompilationContext
from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec


def gen_selective_state_update_module() -> JitSpec:
    return gen_jit_spec(
        "mamba_selective_state_update",
        [
            jit_env.FLASHINFER_CSRC_DIR / "selective_state_update.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mamba_binding.cu",
        ],
    )


def gen_selective_state_update_sm90_module() -> JitSpec:
    # We use a specialized module for Hopper GPUs due to the explicit use
    # of TMA device functions (vertical producer-consumer kernel).
    # This supports SM90 (Hopper) only.
    #
    # Technically, all the kernels in this module can be executed on newer GPUs than Hopper,
    # but this kernel ends up being slower than the alternative SM100 module.
    # Therefore, this is excluded to reduce the amount of compilation.
    compilation_context = CompilationContext()
    nvcc_flags = compilation_context.get_nvcc_flags_list(supported_major_versions=[9])
    nvcc_flags += [
        "-DFLASHINFER_MAMBA_ENABLE_SM90",
    ]

    return gen_jit_spec(
        "mamba_selective_state_update_sm90",
        [
            jit_env.FLASHINFER_CSRC_DIR / "selective_state_update.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mamba_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )


def gen_selective_state_update_sm100_module() -> JitSpec:
    # We use a specialized module for Blackwell+ GPUs with horizontal
    # producer-consumer kernel optimized for SM100 and newer architectures.
    # This supports SM100 (Blackwell) and future architectures.
    # Technically, the code in this module can compile on sm90 as well, but
    # this kernel is a lot slower on hopper than those in the mamba_selective_state_update and
    # mamba_selective_state_update_sm90 modules.
    compilation_context = CompilationContext()
    nvcc_flags = compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11, 12]
    )
    nvcc_flags += [
        "-DFLASHINFER_MAMBA_ENABLE_SM100",
    ]

    return gen_jit_spec(
        "mamba_selective_state_update_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "selective_state_update.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mamba_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )
