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

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec


# The scorer multiplies with m16n8k16, which arrived with compute capability
# 8.0. Every 8.x, 9.x, 10.x and 12.x target has it; 7.x does not.
SPARSE_SCORES_SUPPORTED_MAJOR_VERSIONS = [8, 9, 10, 11, 12]


def gen_sparse_scores_module() -> JitSpec:
    # The device check the caller makes is not enough on its own. Without
    # -gencode flags of its own this module is built for every architecture in
    # the build, so a mixed 7.5 + 8.0 AOT build compiles the m16n8k16 path for
    # 7.5 and nvcc rejects it. The flags are filtered here instead.
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=SPARSE_SCORES_SUPPORTED_MAJOR_VERSIONS
    )
    return gen_jit_spec(
        "sparse_scores",
        [
            jit_env.FLASHINFER_CSRC_DIR / "sparse_scores.cu",
            jit_env.FLASHINFER_CSRC_DIR / "sparse_scores_jit_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16"],
    )
