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

from pathlib import Path

from ...jit import env as jit_env
from ...jit.core import JitSpec, current_compilation_context, gen_jit_spec

# Directory containing msa_ops CUDA source files
_MSA_CSRC_DIR = Path(__file__).resolve().parents[1] / "csrc"


def gen_sparse_topk_select_module() -> JitSpec:
    """JIT module generator for sparse_topk_select targeting SM120/SM121."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )

    return gen_jit_spec(
        "msa_ops_sparse_topk_select",
        sources=[_MSA_CSRC_DIR / "sparse_topk_select.cu"],
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=[
            _MSA_CSRC_DIR / "include",  # sparse_topk_select.cuh
            jit_env.FLASHINFER_CSRC_DIR,  # tvm_ffi_utils.h
            *jit_env.CCCL_INCLUDE_DIRS,  # cub/cub.cuh
        ],
    )
