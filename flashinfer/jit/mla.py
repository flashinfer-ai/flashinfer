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
from .core import JitSpec, gen_jit_spec, current_compilation_context


def gen_mla_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11]
    )
    return gen_jit_spec(
        "mla",
        [
            jit_env.FLASHINFER_CSRC_DIR / "cutlass_mla.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mla_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )
