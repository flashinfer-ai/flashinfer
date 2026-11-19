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


def gen_kda_batch_invariant_verify_module() -> JitSpec:
    """The snapshot and restore of the batch-invariant KDA class's multi-token verify (SM100 / SM103 / SM107)."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )
    # its own module: it holds no row arithmetic, so it stays outside the class's fingerprint sources
    return gen_jit_spec(
        "kda_batch_invariant_verify",
        [jit_env.FLASHINFER_CSRC_DIR / "kda_batch_invariant_verify.cu"],
        extra_cuda_cflags=[*nvcc_flags, "-fmad=false"],
        use_fast_math=False,
    )
