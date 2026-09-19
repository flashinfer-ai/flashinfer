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
from .core import JitSpec, gen_jit_spec


def gen_sparse_scores_module() -> JitSpec:
    # The scorer's mma path is sm_80 and newer; the caller checks the device
    # before asking for this module.
    return gen_jit_spec(
        "sparse_scores",
        [
            jit_env.FLASHINFER_CSRC_DIR / "sparse_scores.cu",
            jit_env.FLASHINFER_CSRC_DIR / "sparse_scores_jit_binding.cu",
        ],
        extra_cuda_cflags=["-DENABLE_BF16"],
    )
