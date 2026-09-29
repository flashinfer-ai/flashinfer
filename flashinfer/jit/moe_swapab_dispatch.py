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
from .core import JitSpec, current_compilation_context, gen_jit_spec


def gen_swapab_dispatch_module() -> JitSpec:
    """Dual-width work-list dispatch for the CuTe DSL swap-AB grouped GEMMs
    (single-CTA scan over the ``moe_sort`` row groups; SM90+ for the PDL
    intrinsics)."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10], map_sm107_to_100f=True
    )
    return gen_jit_spec(
        "moe_swapab_dispatch",
        [jit_env.FLASHINFER_CSRC_DIR / "moe_swapab_dispatch.cu"],
        extra_cuda_cflags=nvcc_flags,
    )
